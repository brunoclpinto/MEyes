//
//  Speak.swift
//  CameraAccess
//
//  Created by Bruno Pinto on 20/02/2026.
//

import AVFoundation

@MainActor
final class Speaker {
    private let synth = AVSpeechSynthesizer()

    /// Resolved once and cached. Picks the best quality voice available
    /// for the current system language (premium > enhanced > default).
    private lazy var resolvedVoice: AVSpeechSynthesisVoice? = {
        let langCode = AVSpeechSynthesisVoice.currentLanguageCode()
        let voices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language == langCode }

        // Prefer highest quality available
        return voices.first(where: { $0.quality == .premium })
            ?? voices.first(where: { $0.quality == .enhanced })
            ?? voices.first
            ?? AVSpeechSynthesisVoice(language: langCode)
    }()

    func speak(_ text: String,
               rate: Float = AVSpeechUtteranceDefaultSpeechRate,
               pitch: Float = 1.0,
               volume: Float = 1.0) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.prefersAssistiveTechnologySettings = true
        utterance.voice = resolvedVoice
        utterance.rate = rate
        utterance.pitchMultiplier = pitch
        utterance.volume = volume
        synth.speak(utterance)
    }

    func stop(immediately: Bool = true) {
        synth.stopSpeaking(at: immediately ? .immediate : .word)
    }
}

import Foundation

extension String {
  func leadingNaturalNumber() -> String {
      let pattern = #"^\s*(\d+)"#   // optional leading spaces, then digits
      guard
          let re = try? NSRegularExpression(pattern: pattern),
          let m = re.firstMatch(in: self, range: NSRange(self.startIndex..., in: self)),
          let r = Range(m.range(at: 1), in: self)
      else { return "" }
    
    guard let number = Int(self[r]) else {
      return ""
    }
    
    return "\(number)"
  }
}
