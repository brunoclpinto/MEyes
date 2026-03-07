import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import Metal

// MARK: - OCRPreset

public struct OCRPreset: Sendable {
    public enum Thin: Sendable {
        case disabled
        /// maxIterations: 0 = until stable
        case enabled(maxIterations: Int = 0)
    }

    public var scale: Float
    public var flatten: Bool
    public var core: Float
    public var binarize: Bool
    public var thresh: Float?     // 0..1, nil => Otsu when binarize=true
    public var shrink: Float
    public var thin: Thin
    public var regrow: Float

    nonisolated public init(
        scale: Float = 1,
        flatten: Bool = true,
        core: Float = 1,
        binarize: Bool = false,
        thresh: Float? = nil,
        shrink: Float = 0,
        thin: Thin = .disabled,
        regrow: Float = 0
    ) {
        self.scale = scale
        self.flatten = flatten
        self.core = core
        self.binarize = binarize
        self.thresh = thresh
        self.shrink = shrink
        self.thin = thin
        self.regrow = regrow
    }

    /// Default preset — grayscale, no binarization.
    public nonisolated static var `default`: OCRPreset {
        OCRPreset(
            scale: 1, flatten: true, core: 1,
            binarize: false, thresh: nil, shrink: 0,
            thin: .disabled, regrow: 0
        )
    }

    /// Minimal grayscale only — all preprocessing disabled.
    public static let disabled = OCRPreset(
        scale: 1, flatten: false, core: 0,
        binarize: false, thresh: nil, shrink: 0,
        thin: .disabled, regrow: 0
    )
}

// MARK: - OCRPipeline

public enum OCRPipeline {

    public static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() { return CIContext(mtlDevice: device) }
        return CIContext()
    }()

    public static func ocrCIImage(
        _ ciInput: CIImage,
        preset: OCRPreset,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) async throws -> String {

        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    var img = ciInput

                    // 1) grayscale
                    img = img.applyingFilter("CIColorControls", parameters: [
                        kCIInputSaturationKey: 0.0,
                        kCIInputContrastKey: 1.0,
                        kCIInputBrightnessKey: 0.0
                    ])

                    // 2) flatten illumination (optional)
                    if preset.flatten {
                        let extent = img.extent
                        let minDim = min(extent.width, extent.height)
                        let radius = Float(max(10.0, Double(minDim * 0.03)))

                        let clamped = img.clampedToExtent()
                        let blur = CIFilter.gaussianBlur()
                        blur.inputImage = clamped
                        blur.radius = radius
                        let blurred = (blur.outputImage ?? clamped).cropped(to: extent)

                        img = img.applyingFilter("CIDivideBlendMode", parameters: [
                            kCIInputBackgroundImageKey: blurred
                        ])
                        img = img.applyingFilter("CIColorControls", parameters: [
                            kCIInputSaturationKey: 0.0,
                            kCIInputContrastKey: 1.4,
                            kCIInputBrightnessKey: 0.0
                        ]).cropped(to: img.extent.integral)
                    }

                    // 3) upscale
                    if preset.scale != 1 {
                        let lz = CIFilter.lanczosScaleTransform()
                        lz.inputImage = img
                        lz.scale = preset.scale
                        lz.aspectRatio = 1.0
                        let out = lz.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // 4) core tighten (pre-threshold)
                    if preset.core >= 1 {
                        let core = CIFilter.morphologyRectangleMaximum()
                        core.inputImage = img
                        core.width = preset.core
                        core.height = preset.core
                        let out = core.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // If not binarizing, feed grayscale to Vision
                    if !preset.binarize {
                        let cgOut = try renderToCGImage(img)
                        let text = try visionOCR(
                            cgImage: cgOut,
                            recognitionLanguages: recognitionLanguages,
                            usesLanguageCorrection: usesLanguageCorrection,
                            recognitionLevel: recognitionLevel
                        )
                        continuation.resume(returning: text)
                        return
                    }

                    // 5) threshold
                    if let t = preset.thresh {
                        let tt = max(0, min(1, t))
                        img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": tt])
                            .cropped(to: img.extent.integral)
                    } else {
                        if #available(iOS 14.0, macOS 12.0, *) {
                            let otsu = CIFilter.colorThresholdOtsu()
                            otsu.inputImage = img
                            let out = otsu.outputImage ?? img
                            img = out.cropped(to: out.extent.integral)
                        } else {
                            img = img.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": 0.5])
                                .cropped(to: img.extent.integral)
                        }
                    }

                    // Normalize to black text on white
                    if meanBrightness(img) < 0.5 {
                        img = img.applyingFilter("CIColorInvert").cropped(to: img.extent.integral)
                    }

                    // 6) shrink
                    if preset.shrink >= 1 {
                        let shrink = CIFilter.morphologyRectangleMaximum()
                        shrink.inputImage = img
                        shrink.width = preset.shrink
                        shrink.height = preset.shrink
                        let out = shrink.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    // 7) thinning
                    switch preset.thin {
                    case .disabled:
                        break
                    case .enabled(let maxIter):
                        img = try thinBinaryCIImage(img, maxIterations: max(0, maxIter))
                    }

                    // 8) regrow
                    if preset.regrow >= 1 {
                        let grow = CIFilter.morphologyRectangleMinimum()
                        grow.inputImage = img
                        grow.width = preset.regrow
                        grow.height = preset.regrow
                        let out = grow.outputImage ?? img
                        img = out.cropped(to: out.extent.integral)
                    }

                    let cgOut = try renderToCGImage(img)
                    let text = try visionOCR(
                        cgImage: cgOut,
                        recognitionLanguages: recognitionLanguages,
                        usesLanguageCorrection: usesLanguageCorrection,
                        recognitionLevel: recognitionLevel
                    )
                    continuation.resume(returning: text)

                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Internals

    static func renderToCGImage(_ img: CIImage) throws -> CGImage {
        guard let cg = ciContext.createCGImage(img, from: img.extent.integral) else {
            throw NSError(domain: "OCRPipeline", code: 100,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot render OCR CIImage"])
        }
        return cg
    }

    static func visionOCR(
        cgImage: CGImage,
        recognitionLanguages: [String]?,
        usesLanguageCorrection: Bool,
        recognitionLevel: VNRequestTextRecognitionLevel
    ) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = recognitionLevel
        request.usesLanguageCorrection = usesLanguageCorrection
        if let langs = recognitionLanguages, !langs.isEmpty {
            request.recognitionLanguages = langs
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let obs = request.results ?? []
        let sorted = obs.sorted {
            if $0.boundingBox.minY != $1.boundingBox.minY {
                return $0.boundingBox.minY > $1.boundingBox.minY
            }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }

        return sorted.compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
    }

    static func meanBrightness(_ image: CIImage) -> Double {
        let avg = CIFilter.areaAverage()
        avg.inputImage = image
        avg.extent = image.extent

        guard let out = avg.outputImage,
              let cg = ciContext.createCGImage(out, from: CGRect(x: 0, y: 0, width: 1, height: 1))
        else { return 0.0 }

        var px = [UInt8](repeating: 0, count: 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        return px.withUnsafeMutableBytes { buf -> Double in
            guard let ctx = CGContext(
                data: buf.baseAddress,
                width: 1, height: 1,
                bitsPerComponent: 8,
                bytesPerRow: 4,
                space: cs,
                bitmapInfo: bitmapInfo
            ) else { return 0.0 }

            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: 1, height: 1))
            let p = buf.bindMemory(to: UInt8.self)
            let b = Double(p[0]), g = Double(p[1]), r = Double(p[2])
            return (r + g + b) / (3.0 * 255.0)
        }
    }

    // MARK: - Zhang–Suen thinning

    static func thinBinaryCIImage(_ img: CIImage, maxIterations: Int) throws -> CIImage {
        let cg = try renderToCGImage(img)
        guard let g = cgImageToGrayBytes(cg) else { return img }

        var bin = [UInt8](repeating: 0, count: g.bytes.count)
        for i in 0..<g.bytes.count { bin[i] = (g.bytes[i] < 128) ? 1 : 0 }

        let thinned = zhangSuenThin(bin, width: g.width, height: g.height, maxIterations: maxIterations)

        var out = [UInt8](repeating: 255, count: thinned.count)
        for i in 0..<thinned.count { out[i] = (thinned[i] == 1) ? 0 : 255 }

        if let cg2 = grayBytesToCGImage(out, width: g.width, height: g.height) {
            return CIImage(cgImage: cg2).cropped(to: CGRect(x: 0, y: 0, width: g.width, height: g.height))
        }
        return img
    }

    static func zhangSuenThin(_ src: [UInt8], width: Int, height: Int, maxIterations: Int) -> [UInt8] {
        var img = src
        if width < 3 || height < 3 { return img }

        func idx(_ x: Int, _ y: Int) -> Int { y * width + x }
        var iter = 0

        while true {
            var changed = false
            var toRemove = [Bool](repeating: false, count: img.count)

            // Step 1
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let p = idx(x, y)
                    if img[p] == 0 { continue }

                    let p2 = img[idx(x, y-1)],   p3 = img[idx(x+1, y-1)]
                    let p4 = img[idx(x+1, y)],    p5 = img[idx(x+1, y+1)]
                    let p6 = img[idx(x, y+1)],    p7 = img[idx(x-1, y+1)]
                    let p8 = img[idx(x-1, y)],    p9 = img[idx(x-1, y-1)]

                    let n = Int(p2+p3+p4+p5+p6+p7+p8+p9)
                    if n < 2 || n > 6 { continue }

                    let s = ((p2==0 && p3==1) ? 1 : 0) + ((p3==0 && p4==1) ? 1 : 0) +
                            ((p4==0 && p5==1) ? 1 : 0) + ((p5==0 && p6==1) ? 1 : 0) +
                            ((p6==0 && p7==1) ? 1 : 0) + ((p7==0 && p8==1) ? 1 : 0) +
                            ((p8==0 && p9==1) ? 1 : 0) + ((p9==0 && p2==1) ? 1 : 0)
                    if s != 1 { continue }

                    if (p2 * p4 * p6) != 0 { continue }
                    if (p4 * p6 * p8) != 0 { continue }

                    toRemove[p] = true
                }
            }
            for i in 0..<toRemove.count where toRemove[i] { img[i] = 0; changed = true }

            toRemove = [Bool](repeating: false, count: img.count)

            // Step 2
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let p = idx(x, y)
                    if img[p] == 0 { continue }

                    let p2 = img[idx(x, y-1)],   p3 = img[idx(x+1, y-1)]
                    let p4 = img[idx(x+1, y)],    p5 = img[idx(x+1, y+1)]
                    let p6 = img[idx(x, y+1)],    p7 = img[idx(x-1, y+1)]
                    let p8 = img[idx(x-1, y)],    p9 = img[idx(x-1, y-1)]

                    let n = Int(p2+p3+p4+p5+p6+p7+p8+p9)
                    if n < 2 || n > 6 { continue }

                    let s = ((p2==0 && p3==1) ? 1 : 0) + ((p3==0 && p4==1) ? 1 : 0) +
                            ((p4==0 && p5==1) ? 1 : 0) + ((p5==0 && p6==1) ? 1 : 0) +
                            ((p6==0 && p7==1) ? 1 : 0) + ((p7==0 && p8==1) ? 1 : 0) +
                            ((p8==0 && p9==1) ? 1 : 0) + ((p9==0 && p2==1) ? 1 : 0)
                    if s != 1 { continue }

                    if (p2 * p4 * p8) != 0 { continue }
                    if (p2 * p6 * p8) != 0 { continue }

                    toRemove[p] = true
                }
            }
            for i in 0..<toRemove.count where toRemove[i] { img[i] = 0; changed = true }

            iter += 1
            if !changed { break }
            if maxIterations > 0 && iter >= maxIterations { break }
        }

        return img
    }

    static func cgImageToGrayBytes(_ cg: CGImage) -> (bytes: [UInt8], width: Int, height: Int)? {
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h)

        let ok = buf.withUnsafeMutableBytes { ptr -> Bool in
            guard let ctx = CGContext(
                data: ptr.baseAddress,
                width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? (buf, w, h) : nil
    }

    static func grayBytesToCGImage(_ bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        var data = bytes
        guard let provider = CGDataProvider(data: Data(bytes: &data, count: data.count) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
