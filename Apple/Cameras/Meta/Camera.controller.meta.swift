//
//  Camera.controller.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

import MWDATCore

public actor CameraControllerMeta: CameraController {
  public private(set) var cameras: [any Camera] = []

  public init() async {
    try? Wearables.configure()
    let wearables = Wearables.shared
    for deviceId in wearables.devices {
      cameras.append(CameraMeta(deviceId: deviceId, wearables: wearables))
    }
  }
}
