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

package mig

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
)

const (
	nvidiaCapDir = "/proc/driver/nvidia/capabilities"
)

// DeviceManager performs various management operations on mig devices.
type DeviceManager struct {
	devDirectory  string
	procDirectory string
	nvidiaSmiPath string
}

// NewDeviceManager creates a new DeviceManager to handle MIG devices on the node.
func NewDeviceManager(devDirectory, procDirectory, nvidiaSmiPath string) *DeviceManager {
	return &DeviceManager{
		devDirectory:  devDirectory,
		procDirectory: procDirectory,
		nvidiaSmiPath: nvidiaSmiPath,
	}
}

// CurrentMigMode returns whether mig mode is currently enabled all GPUs attached to this node.
func (d *DeviceManager) CurrentMigMode() (bool, error) {
	out, err := exec.Command(d.nvidiaSmiPath, "--query-gpu=mig.mode.current", "--format=csv,noheader").Output()
	if err != nil {
		return false, fmt.Errorf("unable to execute nvidia-smi to query current mig mode: %v", err)
	}
	if strings.HasPrefix(string(out), "Enabled") {
		return true, nil
	}
	if strings.HasPrefix(string(out), "Disabled") {
		return false, nil
	}
	return false, fmt.Errorf("unable to check if mig mode is enabled: nvidia-smi returned invalid output: %s", out)
}

// EnableMigMode enables mig mode on all GPUs attached to the node that already do not have mig mode enabled.
func (d *DeviceManager) EnableMigMode() error {
	return exec.Command(d.nvidiaSmiPath, "-mig", "1").Run()
}

// CreateGPUInstances partitions each GPU on the node into `gpuPartitionCount` equal sized GPU instances.
func (d *DeviceManager) CreateGPUInstances(gpuPartitionCount int) error {
	if err := d.DestroyAllGPUInstance(); err != nil {
		return fmt.Errorf("unable to destroy GPU instances: %v", err)
	}

	p, err := BuildPartitionStr(gpuPartitionCount)
	if err != nil {
		return err
	}
	out, err := exec.Command(d.nvidiaSmiPath, "mig", "-cgi", p).Output()
	if err != nil {
		return fmt.Errorf("failed to create GPU Instances: output: %s, error: %v", out, err)
	}

	out, err = exec.Command(d.nvidiaSmiPath, "mig", "-cci").Output()
	if err != nil {
		return fmt.Errorf("failed to create compute instances: output: %s, error: %v", out, err)
	}

	return nil
}

// DestroyAllGPUInstance destroys all compute and GPU instances on the node
func (d *DeviceManager) DestroyAllGPUInstance() error {
	out, err := exec.Command(d.nvidiaSmiPath, "mig", "-dci").Output()
	if err != nil {
		if strings.Contains(err.Error(), "No GPU instances found") {
			return nil
		}
		return fmt.Errorf("unable to destroy compute instance, output: %serror: %v ", out, err)
	}

	out, err = exec.Command(d.nvidiaSmiPath, "mig", "-dgi").Output()
	if err != nil {
		return fmt.Errorf("unable to destroy gpu instance, output: %serror: %v ", out, err)
	}
	return nil
}

// DiscoverGPUInstance finds all the GPU instances on the node, and returns a list of DeviceSpec for each GPU instance
// that provide access to the GPU Instance.
func (d *DeviceManager) DiscoverGPUInstance() (map[string][]pluginapi.DeviceSpec, error) {
	ret := make(map[string][]pluginapi.DeviceSpec)
	capFiles, err := ioutil.ReadDir(nvidiaCapDir)
	if err != nil {
		return ret, fmt.Errorf("failed to read capabilities directory: %v", err)
	}

	gpuFileRegexp := regexp.MustCompile("gpu([0-9]+)")
	giFileRegexp := regexp.MustCompile("gi([0-9]+)")
	deviceRegexp := regexp.MustCompile("DeviceFileMinor: ([0-9]+)")
	for _, capFile := range capFiles {
		m := gpuFileRegexp.FindStringSubmatch(capFile.Name())
		if len(m) != 2 {
			// Not a gpu, continue to next file
			continue
		}

		gpuID := m[1]

		giBasePath := path.Join(nvidiaCapDir, capFile.Name(), "mig")
		giFiles, err := ioutil.ReadDir(giBasePath)
		if err != nil {
			return ret, fmt.Errorf("unable to discover gpu instance: %v", err)
		}

		for _, giFile := range giFiles {
			if !giFileRegexp.MatchString(giFile.Name()) {
				continue
			}

			gpuInstanceID := "nvidia" + gpuID + "/" + giFile.Name()

			giAccessFile := path.Join(giBasePath, giFile.Name(), "access")
			content, err := ioutil.ReadFile(giAccessFile)
			if err != nil {
				return ret, fmt.Errorf("unable to read GPU Instance access file (%s): %v", giAccessFile, err)
			}

			m := deviceRegexp.FindStringSubmatch(string(content))
			if len(m) != 2 {
				return ret, fmt.Errorf("unexpected contents in GPU instance access file(%s): %v", giAccessFile, err)
			}
			giMinorDevice := m[1]

			ciAccessFile := path.Join(giBasePath, giFile.Name(), "ci0", "access")
			content, err = ioutil.ReadFile(ciAccessFile)
			if err != nil {
				return ret, fmt.Errorf("unable to read Compute Instance access file (%s): %v", ciAccessFile, err)
			}

			m = deviceRegexp.FindStringSubmatch(string(content))
			if len(m) != 2 {
				return ret, fmt.Errorf("unexpected contents in compute instance access file(%s): %v", ciAccessFile, err)
			}
			ciMinorDevice := m[1]

			gpuDevice := path.Join(d.devDirectory, "nvidia"+gpuID)
			if _, err := os.Stat(gpuDevice); err != nil {
				return ret, fmt.Errorf("GPU device (%s) not fount: %v", gpuDevice, err)
			}

			giDevice := path.Join(d.devDirectory, "nvidia-caps", "nvidia-cap"+giMinorDevice)
			if _, err := os.Stat(giDevice); err != nil {
				return ret, fmt.Errorf("GPU instance device (%s) not fount: %v", giDevice, err)
			}

			ciDevice := path.Join(d.devDirectory, "nvidia-caps", "nvidia-cap"+ciMinorDevice)
			if _, err := os.Stat(ciDevice); err != nil {
				return ret, fmt.Errorf("GPU instance device (%s) not fount: %v", ciDevice, err)
			}

			ret[gpuInstanceID] = []pluginapi.DeviceSpec{
				{
					ContainerPath: gpuDevice,
					HostPath:      gpuDevice,
					Permissions:   "mrw",
				},
				{
					ContainerPath: giDevice,
					HostPath:      giDevice,
					Permissions:   "mrw",
				},
				{
					ContainerPath: ciDevice,
					HostPath:      ciDevice,
					Permissions:   "mrw",
				},
			}

		}
	}

	return ret, nil
}

// BuildPartitionStr builds a string that represents a partitioning a GPU into `gpuPartitionCount` equal sized GPU instances.
func BuildPartitionStr(gpuPartitionCount int) (string, error) {
	var gpuPartition string
	var err error
	switch gpuPartitionCount {
	case 7:
		gpuPartition = "19,19,19,19,19,19,19"
	default:
		err = fmt.Errorf("invalid gpuPartitionCount: %d", gpuPartitionCount)
	}

	return gpuPartition, err
}
