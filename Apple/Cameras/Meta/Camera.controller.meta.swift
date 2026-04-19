//
//  Camera.controller.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

internal import UIKit
import MWDATCore

public actor CameraControllerMeta: CameraController {
  public private(set) var cameras: [any Camera] = []

  public init() async {
    // 1 — Check whether the Meta AI companion app is installed.
    let metaAIInstalled = await MainActor.run {
      UIApplication.shared.canOpenURL(URL(string: "fb-viewapp://")!)
    }
    guard metaAIInstalled else { return }

    let wearables = Wearables.shared

    // 2 — Already registered: expose a CameraMeta that uses AutoDeviceSelector
    //     to handle device discovery and BLE readiness internally.
    if wearables.registrationState == .registered {
      cameras.append(CameraMeta(wearables: wearables))
      return
    }

    // 3 — Not registered: expose a placeholder camera that drives the registration flow.
    cameras.append(CameraMetaRegistration(wearables: wearables))
  }
}
