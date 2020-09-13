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
	"os/exec"
	"strings"
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
	p, err := BuildPartitionStr(gpuPartitionCount)
	if err != nil {
		return err
	}
	err = exec.Command(d.nvidiaSmiPath, "mig", "-cgi", p).Run()
	if err != nil {
		return fmt.Errorf("failed to create GPU Instances: %v", err)
	}

	err = exec.Command(d.nvidiaSmiPath, "mig", "-cci").Run()
	if err != nil {
		return fmt.Errorf("failed to create compute instances: %s", err)
	}

	return nil
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
