//
//  CoreVideo.extension.swift
//  HackEye
//
//  Created by Bruno Pinto on 06/03/2026.
//

import CoreVideo
import UIKit

public extension UIImage {
  func imageBuffer() -> CVImageBuffer? {
    guard let cgImage = self.cgImage else { return nil }
    
    let width = cgImage.width
    let height = cgImage.height
    
    let attrs: [CFString: Any] = [
      kCVPixelBufferCGImageCompatibilityKey: true,
      kCVPixelBufferCGBitmapContextCompatibilityKey: true
    ]
    
    var buffer: CVImageBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32BGRA,
      attrs as CFDictionary,
      &buffer
    )
    
    guard status == kCVReturnSuccess, let imageBuffer = buffer else {
      return nil
    }
    
    CVPixelBufferLockBaseAddress(imageBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(imageBuffer, []) }
    
    guard let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer) else {
      return nil
    }
    
    let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo =
    CGBitmapInfo.byteOrder32Little.rawValue |
    CGImageAlphaInfo.premultipliedFirst.rawValue
    
    guard let context = CGContext(
      data: baseAddress,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      return nil
    }
    
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return imageBuffer
  }
}
