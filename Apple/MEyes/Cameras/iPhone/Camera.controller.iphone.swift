//
//  Camera.controller.iphone.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

import AVFoundation

public actor CameraControllerIphone: CameraController {
  public private(set) var cameras: [any Camera] = []
  
  public init() async {
    let devices = allLocalVideoCameras()
    for device in devices {
      await device.setFormatAndFPS(fps: 30)
      cameras.append(await CameraIphone(device: device))
    }
  }
  
  private func allLocalVideoCameras() -> [AVCaptureDevice] {
    let types: [AVCaptureDevice.DeviceType] = [
      .builtInWideAngleCamera,
      .builtInUltraWideCamera,
      .builtInTelephotoCamera
    ]
    
    let discovery = AVCaptureDevice.DiscoverySession(
      deviceTypes: types,
      mediaType: .video,
      position: .back
    )
    
    return discovery.devices
  }
}
