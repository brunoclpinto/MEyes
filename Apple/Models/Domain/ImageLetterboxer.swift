import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import Metal

// MARK: - Box

public struct Box: Sendable {
    public var x1: Double
    public var y1: Double
    public var x2: Double
    public var y2: Double

    public var w: Double { max(0, x2 - x1) }
    public var h: Double { max(0, y2 - y1) }
    public var area: Double { w * h }

    public init(x1: Double, y1: Double, x2: Double, y2: Double) {
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
    }

    public func rectXYWH() -> CGRect {
        CGRect(x: x1, y: y1, width: w, height: h)
    }
}

// MARK: - LetterboxMeta

/// Metadata for mapping boxes between original ↔ detector coordinate spaces (top-left origin).
public struct LetterboxMeta: Sendable {
    public let srcW: Double
    public let srcH: Double
    public let dstW: Double
    public let dstH: Double
    public let scale: Double
    public let padX: Double
    public let padY: Double

    public init(srcW: Double, srcH: Double, dstW: Double, dstH: Double,
                scale: Double, padX: Double, padY: Double) {
        self.srcW = srcW; self.srcH = srcH; self.dstW = dstW; self.dstH = dstH
        self.scale = scale; self.padX = padX; self.padY = padY
    }

    /// Map a box from dst (detector input) space back to src (original/crop) space.
    public func dstToSrc(_ b: Box) -> Box {
        let inv = 1.0 / max(scale, 1e-9)
        let x1 = (b.x1 - padX) * inv
        let x2 = (b.x2 - padX) * inv
        let y1 = (b.y1 - padY) * inv
        let y2 = (b.y2 - padY) * inv
        return Box(x1: x1, y1: y1, x2: x2, y2: y2)
    }

    public func clampToSrc(_ b: Box) -> Box {
        let x1 = max(0, min(srcW, b.x1))
        let x2 = max(0, min(srcW, b.x2))
        let y1 = max(0, min(srcH, b.y1))
        let y2 = max(0, min(srcH, b.y2))
        return Box(x1: min(x1, x2), y1: min(y1, y2), x2: max(x1, x2), y2: max(y1, y2))
    }
}

// MARK: - ImageLetterboxer

public enum ImageLetterboxer {

    public static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    /// Letterbox src into dst dimensions, returning the padded image and coordinate metadata.
    public static func letterboxWithMeta(
        _ src: CIImage,
        srcW: Double,
        srcH: Double,
        dstW: Double,
        dstH: Double
    ) -> (CIImage, LetterboxMeta) {
        let scale = min(dstW / max(srcW, 1e-9), dstH / max(srcH, 1e-9))
        let resizedW = srcW * scale
        let resizedH = srcH * scale
        let padX = (dstW - resizedW) / 2.0
        let padY = (dstH - resizedH) / 2.0

        let meta = LetterboxMeta(srcW: srcW, srcH: srcH, dstW: dstW, dstH: dstH,
                                 scale: scale, padX: padX, padY: padY)

        let dstRect = CGRect(x: 0, y: 0, width: dstW, height: dstH)
        let scaled = src.transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
        let dx = CGFloat(padX) - scaled.extent.origin.x
        let dy = CGFloat(padY) - scaled.extent.origin.y
        let translated = scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))

        let bg = CIImage(color: .black).cropped(to: dstRect)
        let out = translated.composited(over: bg).cropped(to: dstRect)

        return (out, meta)
    }

    public static func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let out = pb else {
            throw NSError(domain: "ImageLetterboxer", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot create CVPixelBuffer"])
        }
        return out
    }

    public static func render(_ image: CIImage, to pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let rect = CGRect(x: 0, y: 0,
                          width: CVPixelBufferGetWidth(pb),
                          height: CVPixelBufferGetHeight(pb))
        ciContext.render(image, to: pb, bounds: rect, colorSpace: CGColorSpaceCreateDeviceRGB())
    }
}
