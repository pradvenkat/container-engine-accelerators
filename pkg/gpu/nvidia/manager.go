// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nvidia

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"regexp"
	"strconv"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/container-engine-accelerators/pkg/gpu/nvidia/mig"
	"github.com/golang/glog"
	"google.golang.org/grpc"

	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	// proc directory is used to lookup the access files for each GPU partition.
	procDir = "/proc"
	// All NVIDIA GPUs cards should be mounted with nvidiactl and nvidia-uvm
	// If the driver installed correctly, these two devices will be there.
	nvidiaCtlDevice = "nvidiactl"
	nvidiaUVMDevice = "nvidia-uvm"
	// Optional device.
	nvidiaUVMToolsDevice      = "nvidia-uvm-tools"
	nvidiaDeviceRE            = `^nvidia[0-9]*$`
	gpuCheckInterval          = 10 * time.Second
	pluginSocketCheckInterval = 1 * time.Second
)

var (
	gpuResourceName  = "nvidia.com/gpu"
	vgpuResourceName = "cloud.google.com/vgpu"
)

// GPUConfig stores the settings used to configure the GPUs on a node.
type GPUConfig struct {
	GPUPartitionSize string
	VGPUCountPerGPU  int
}

// nvidiaGPUManager manages nvidia gpu devices.
type nvidiaGPUManager struct {
	devDirectory        string
	mountPaths          []MountPath
	defaultDevices      []string
	devices             map[string]pluginapi.Device
	grpcServer          *grpc.Server
	socket              string
	stop                chan bool
	devicesMutex        sync.Mutex
	nvidiaCtlDevicePath string
	nvidiaUVMDevicePath string
	gpuConfig           GPUConfig
	migDeviceManager    mig.DeviceManager
	Health              chan pluginapi.Device
}

type MountPath struct {
	HostPath      string
	ContainerPath string
}

func validateGPUConfig(gpuConfig GPUConfig) error {
	if gpuConfig.GPUPartitionSize != "" && gpuConfig.VGPUCountPerGPU > 0 {
		return fmt.Errorf("vGPUs and MIG partitions are not supported together at this time")
	}

	return nil
}
func NewNvidiaGPUManager(devDirectory string, mountPaths []MountPath, gpuConfig GPUConfig) *nvidiaGPUManager {
	if err := validateGPUConfig(gpuConfig); err != nil {
		// TODO: change the return type to include error
		return nil
	}

	return &nvidiaGPUManager{
		devDirectory:        devDirectory,
		mountPaths:          mountPaths,
		devices:             make(map[string]pluginapi.Device),
		stop:                make(chan bool),
		nvidiaCtlDevicePath: path.Join(devDirectory, nvidiaCtlDevice),
		nvidiaUVMDevicePath: path.Join(devDirectory, nvidiaUVMDevice),
		gpuConfig:           gpuConfig,
		migDeviceManager:    mig.NewDeviceManager(devDirectory, procDir),
		Health:              make(chan pluginapi.Device),
	}
}

// ListDevices lists all GPU devices (including partitions) available on this node.
func (ngm *nvidiaGPUManager) ListDevices() map[string]pluginapi.Device {
	if ngm.gpuConfig.GPUPartitionSize == "" {
		if ngm.gpuConfig.VGPUCountPerGPU > 0 {
			vgpuCount := len(ngm.devices) * ngm.gpuConfig.VGPUCountPerGPU
			devices := make(map[string]pluginapi.Device, vgpuCount)

			for i := 0; i < vgpuCount; i++ {
				deviceID := fmt.Sprintf("vgpu%d", i)
				devices[deviceID] = pluginapi.Device{
					ID:     deviceID,
					Health: pluginapi.Healthy,
				}
			}

			glog.Infof("vGPU devices disovered: %v", devices)
			return devices
		} else {
			return ngm.devices
		}
	}

	return ngm.migDeviceManager.ListGPUPartitionDevices()
}

func (ngm *nvidiaGPUManager) virtualToPhysicalDeviceID(deviceID string) (string, error) {
	vgpuRegex := regexp.MustCompile("vgpu([0-9]+)$")
	m := vgpuRegex.FindStringSubmatch(deviceID)
	if len(m) != 2 {
		return "", fmt.Errorf("device ID (%s) is not a valid vGPU", deviceID)
	}

	vgpuID, err := strconv.Atoi(m[1])
	if err != nil {
		return "", fmt.Errorf("device ID (%s) is not a valid vGPU: %v", deviceID, err)
	}

	// TODO: check that physical GPU ID does not exceed # of GPUs on the node
	return fmt.Sprintf("nvidia%d", vgpuID/ngm.gpuConfig.VGPUCountPerGPU), nil
}

// DeviceSpec returns the device spec that inclues list of devices to allocate for a deviceID.
func (ngm *nvidiaGPUManager) DeviceSpec(deviceID string) ([]pluginapi.DeviceSpec, error) {
	if ngm.gpuConfig.GPUPartitionSize == "" {
		if ngm.gpuConfig.VGPUCountPerGPU > 0 {
			var err error
			deviceID, err = ngm.virtualToPhysicalDeviceID(deviceID)
			if err != nil {
				return nil, err
			}
		}

		deviceSpecs := make([]pluginapi.DeviceSpec, 0)
		dev, ok := ngm.devices[deviceID]
		if !ok {
			return deviceSpecs, fmt.Errorf("invalid allocation request with non-existing device %s", deviceID)
		}
		if dev.Health != pluginapi.Healthy {
			return deviceSpecs, fmt.Errorf("invalid allocation request with unhealthy device %s", deviceID)
		}
		deviceSpecs = append(deviceSpecs, pluginapi.DeviceSpec{
			HostPath:      path.Join(ngm.devDirectory, deviceID),
			ContainerPath: path.Join(ngm.devDirectory, deviceID),
			Permissions:   "mrw",
		})
		return deviceSpecs, nil
	}
	return ngm.migDeviceManager.DeviceSpec(deviceID)
}

// Discovers all NVIDIA GPU devices available on the local node by walking nvidiaGPUManager's devDirectory.
func (ngm *nvidiaGPUManager) discoverGPUs() error {
	reg := regexp.MustCompile(nvidiaDeviceRE)
	files, err := ioutil.ReadDir(ngm.devDirectory)
	if err != nil {
		return err
	}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if reg.MatchString(f.Name()) {
			glog.V(3).Infof("Found Nvidia GPU %q\n", f.Name())
			ngm.SetDeviceHealth(f.Name(), pluginapi.Healthy)
		}
	}
	return nil
}

func (ngm *nvidiaGPUManager) hasAdditionalGPUsInstalled() bool {
	ngm.devicesMutex.Lock()
	originalDeviceCount := len(ngm.devices)
	ngm.devicesMutex.Unlock()
	deviceCount, err := ngm.discoverNumGPUs()
	if err != nil {
		glog.Errorln(err)
		return false
	}
	if deviceCount > originalDeviceCount {
		glog.Infof("Found %v GPUs, while only %v are registered. Stopping device-plugin server.", deviceCount, originalDeviceCount)
		return true
	}
	return false
}

func (ngm *nvidiaGPUManager) discoverNumGPUs() (int, error) {
	reg := regexp.MustCompile(nvidiaDeviceRE)
	deviceCount := 0
	files, err := ioutil.ReadDir(ngm.devDirectory)
	if err != nil {
		return deviceCount, err
	}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if reg.MatchString(f.Name()) {
			deviceCount++
		}
	}
	return deviceCount, nil
}

// SetDeviceHealth sets the health status for a GPU device or partition if MIG is enabled
func (ngm *nvidiaGPUManager) SetDeviceHealth(name string, health string) {
	ngm.devicesMutex.Lock()
	defer ngm.devicesMutex.Unlock()

	reg := regexp.MustCompile(nvidiaDeviceRE)
	if reg.MatchString(name) {
		ngm.devices[name] = pluginapi.Device{ID: name, Health: health}
	} else {
		ngm.migDeviceManager.SetDeviceHealth(name, health)
	}
}

// Checks if the two nvidia paths exist. Could be used to verify if the driver
// has been installed correctly
func (ngm *nvidiaGPUManager) CheckDevicePaths() error {
	if _, err := os.Stat(ngm.nvidiaCtlDevicePath); err != nil {
		return err
	}

	if _, err := os.Stat(ngm.nvidiaUVMDevicePath); err != nil {
		return err
	}
	return nil
}

// Discovers Nvidia GPU devices and sets up device access environment.
func (ngm *nvidiaGPUManager) Start() error {
	if err := ngm.CheckDevicePaths(); err != nil {
		return fmt.Errorf("error checking device paths: %v", err)
	}
	ngm.defaultDevices = []string{ngm.nvidiaCtlDevicePath, ngm.nvidiaUVMDevicePath}

	nvidiaUVMToolsDevicePath := path.Join(ngm.devDirectory, nvidiaUVMToolsDevice)
	if _, err := os.Stat(nvidiaUVMToolsDevicePath); err == nil {
		ngm.defaultDevices = append(ngm.defaultDevices, nvidiaUVMToolsDevicePath)
	}

	if err := ngm.discoverGPUs(); err != nil {
		return err
	}
	if ngm.gpuConfig.GPUPartitionSize != "" {
		if err := ngm.migDeviceManager.Start(ngm.gpuConfig.GPUPartitionSize); err != nil {
			return fmt.Errorf("failed to start mig device manager: %v", err)
		}
	}

	return nil
}

func (ngm *nvidiaGPUManager) Serve(pMountPath, kEndpoint, pluginEndpoint string) {
	registerWithKubelet := false
	if _, err := os.Stat(path.Join(pMountPath, kEndpoint)); err == nil {
		glog.Infof("will use alpha API\n")
		registerWithKubelet = true
	} else {
		glog.Infof("will use beta API\n")
	}

	for {
		select {
		case <-ngm.stop:
			close(ngm.stop)
			return
		default:
			{
				pluginEndpointPath := path.Join(pMountPath, pluginEndpoint)
				glog.Infof("starting device-plugin server at: %s\n", pluginEndpointPath)
				lis, err := net.Listen("unix", pluginEndpointPath)
				if err != nil {
					glog.Fatalf("starting device-plugin server failed: %v", err)
				}
				ngm.socket = pluginEndpointPath
				ngm.grpcServer = grpc.NewServer()

				// Registers the supported versions of service.
				pluginalpha := &pluginServiceV1Alpha{ngm: ngm}
				pluginalpha.RegisterService()
				pluginbeta := &pluginServiceV1Beta1{ngm: ngm}
				pluginbeta.RegisterService()

				var wg sync.WaitGroup
				wg.Add(1)
				// Starts device plugin service.
				go func() {
					defer wg.Done()
					// Blocking call to accept incoming connections.
					err := ngm.grpcServer.Serve(lis)
					glog.Errorf("device-plugin server stopped serving: %v", err)
				}()

				if registerWithKubelet {
					// Wait till the grpcServer is ready to serve services.
					for len(ngm.grpcServer.GetServiceInfo()) <= 0 {
						time.Sleep(1 * time.Second)
					}
					glog.Infoln("device-plugin server started serving")
					// Registers with Kubelet.
					resourceName := gpuResourceName
					if ngm.gpuConfig.VGPUCountPerGPU > 0 {
						resourceName = vgpuResourceName
					}
					err = RegisterWithKubelet(path.Join(pMountPath, kEndpoint), pluginEndpoint, resourceName)
					if err != nil {
						glog.Infoln("falling back to v1beta1 API")
						err = RegisterWithV1Beta1Kubelet(path.Join(pMountPath, kEndpoint), pluginEndpoint, resourceName)
					}
					if err != nil {
						ngm.grpcServer.Stop()
						wg.Wait()
						glog.Fatal(err)
					}
					glog.Infoln("device-plugin registered with the kubelet")
				}

				// This is checking if the plugin socket was deleted
				// and also if there are additional GPU devices installed.
				// If so, stop the grpc server and start the whole thing again.
				gpuCheck := time.NewTicker(gpuCheckInterval)
				pluginSocketCheck := time.NewTicker(pluginSocketCheckInterval)
				defer gpuCheck.Stop()
				defer pluginSocketCheck.Stop()
			statusCheck:
				for {
					select {
					case <-pluginSocketCheck.C:
						if _, err := os.Lstat(pluginEndpointPath); err != nil {
							glog.Infof("stopping device-plugin server at: %s\n", pluginEndpointPath)
							glog.Errorln(err)
							ngm.grpcServer.Stop()
							break statusCheck
						}
					case <-gpuCheck.C:
						if ngm.hasAdditionalGPUsInstalled() {
							ngm.grpcServer.Stop()
							for {
								err := ngm.discoverGPUs()
								if err == nil {
									break statusCheck
								}
							}
						}

					}
				}
				wg.Wait()
			}
		}
	}
}

func (ngm *nvidiaGPUManager) Stop() error {
	glog.Infof("removing device plugin socket %s\n", ngm.socket)
	if err := os.Remove(ngm.socket); err != nil && !os.IsNotExist(err) {
		return err
	}
	ngm.stop <- true
	<-ngm.stop
	close(ngm.Health)
	return nil
}
