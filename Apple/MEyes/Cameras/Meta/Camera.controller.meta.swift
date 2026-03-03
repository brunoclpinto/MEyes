//
//  Camera.controller.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

import MWDATCore
import Foundation

public actor CameraControllerMeta: CameraController {
  public private(set) var cameras: [any Camera] = []
  
//  private let wearables = Wearables.shared
//  private var devices: [DeviceIdentifier] = Wearables.shared.devices
  
  public init() async {
  }
}
