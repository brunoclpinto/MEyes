//
//  HackEyeApp.swift
//  HackEye
//
//  Created by Bruno Pinto on 23/02/2026.
//

import SwiftUI
import MWDATCore

@main
struct HackEyeApp: App {
    init() {
        do {
            try Wearables.configure()
            print("[HackEye] Wearables SDK configured successfully")
        } catch {
            print("[HackEye] Wearables.configure() failed: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            MainView()
                .onOpenURL { url in
                    guard
                        let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                        components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
                    else { return }
                    Task {
                        _ = try? await Wearables.shared.handleUrl(url)
                    }
                }
        }
    }
}
