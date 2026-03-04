//
//  AVCaptureDevice.extensions.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import AVFoundation

public extension AVCaptureDevice.DeviceType {
  var humanReadable: String {
    switch self {
      case .builtInWideAngleCamera:
        return "Wide Angle".localizedCapitalized
      case .builtInTelephotoCamera:
        return "Telephoto".localizedCapitalized
      case .builtInUltraWideCamera:
        return "Ultra Wide".localizedCapitalized
      default:
        return "Unknown".localizedCapitalized
    }
  }
  
  var zoom: String {
    switch self {
      case .builtInWideAngleCamera:
        return "(1x)"
      case .builtInTelephotoCamera:
        return "(2.5x)"
      case .builtInUltraWideCamera:
        return "(0.5x)"
      default:
        return "?x"
    }
  }
}

public extension AVCaptureDevice {
  func setFormatAndFPS(fps: Double) {
    guard
      let format = self.formats.first(where: { format in
      format.videoSupportedFrameRateRanges.contains { r in
        r.minFrameRate <= fps && fps <= r.maxFrameRate
      }
    })
    else {
      return
    }
    
    try? self.lockForConfiguration()
    defer { self.unlockForConfiguration() }
    
    self.activeFormat = format
    let duration = CMTime(value: 1, timescale: Int32(fps.rounded()))
    self.activeVideoMinFrameDuration = duration
    self.activeVideoMaxFrameDuration = duration
  }
}
