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

    // 2 — Already registered: discover real devices.
    if wearables.registrationState == .registered {
      let discovered = await Self.waitForDevices(wearables: wearables, timeout: .seconds(15))
      for deviceId in discovered {
        cameras.append(CameraMeta(deviceId: deviceId, wearables: wearables))
      }
      return
    }

    // 3 — Not registered: expose a placeholder camera that drives the registration flow.
    cameras.append(CameraMetaRegistration(wearables: wearables))
  }

  // MARK: - Static helpers

  static func waitForDevices(
    wearables: WearablesInterface,
    timeout: Duration
  ) async -> [DeviceIdentifier] {
    let existing = wearables.devices
    if !existing.isEmpty { return existing }

    return await withTaskGroup(of: [DeviceIdentifier].self) { group in
      group.addTask {
        for await devices in wearables.devicesStream() {
          if !devices.isEmpty { return devices }
        }
        return []
      }
      group.addTask {
        try? await Task.sleep(for: timeout)
        return []
      }
      let result = await group.next() ?? []
      group.cancelAll()
      return result
    }
  }
}
