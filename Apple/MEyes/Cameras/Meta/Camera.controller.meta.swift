//
//  Camera.controller.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

import Foundation

public actor CameraControllerMeta: CameraController {
  public private(set) var cameras: [any Camera] = []
  
  public init() async {
  }
}
