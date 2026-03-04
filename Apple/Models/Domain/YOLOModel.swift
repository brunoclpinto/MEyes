import Foundation
import CoreML

// MARK: - YOLODetection

public struct YOLODetection: Sendable {
    public var x1: Double
    public var y1: Double
    public var x2: Double
    public var y2: Double
    public var score: Double
    public var cls: Int
    public var area: Double { max(0, x2 - x1) * max(0, y2 - y1) }

    public init(x1: Double, y1: Double, x2: Double, y2: Double, score: Double, cls: Int) {
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        self.score = score; self.cls = cls
    }
}

// MARK: - YOLOModel

public final class YOLOModel: @unchecked Sendable {

    public enum Source {
        case bundle(name: String, bundle: Bundle = .main)
        case mlModel(MLModel)
        case compiledURL(URL)
    }

    public enum BoxOrigin: Sendable {
        case topLeft
        case bottomLeft
    }

    public let model: MLModel
    public let inputName: String
    public let outputName: String
    public let boxOrigin: BoxOrigin

    public init(_ source: Source, boxOrigin: BoxOrigin = .topLeft, computeUnits: MLComputeUnits = .all) throws {
        self.boxOrigin = boxOrigin

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        switch source {
        case .bundle(let name, let bundle):
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
                throw NSError(domain: "YOLOModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(name).mlmodelc in bundle"])
            }
            self.model = try MLModel(contentsOf: url, configuration: cfg)

        case .mlModel(let m):
            self.model = m

        case .compiledURL(let url):
            self.model = try MLModel(contentsOf: url, configuration: cfg)
        }

        let inputs = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard !inputs.isEmpty else {
            throw NSError(domain: "YOLOModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model has no inputs"])
        }
        self.inputName = inputs.contains("image") ? "image" : inputs[0]

        let outputs = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard !outputs.isEmpty else {
            throw NSError(domain: "YOLOModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Model has no outputs"])
        }
        self.outputName = outputs[0]
    }

    public func predict(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
        let out = try model.prediction(from: provider)
        guard let fv = out.featureValue(for: outputName), let arr = fv.multiArrayValue else {
            throw NSError(domain: "YOLOModel", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output \(outputName) is not MLMultiArray"])
        }
        return arr
    }

    // MARK: - Detection parsing

    public static func parseDetections(_ arr: MLMultiArray, inputW: Double, inputH: Double) -> [YOLODetection] {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }

        func offset(_ idx: [Int]) -> Int {
            var o = 0
            for (i, s) in zip(idx, strides) { o += i * s }
            return o
        }

        let rank = shape.count
        var N = 0
        var get: (_ i: Int, _ j: Int) -> Double = { _, _ in 0 }

        if rank == 3, shape[2] == 6 {
            N = shape[1]
            get = { i, j in readMultiArrayValue(arr, linearIndex: offset([0, i, j])) }
        } else if rank == 2, shape[1] == 6 {
            N = shape[0]
            get = { i, j in readMultiArrayValue(arr, linearIndex: offset([i, j])) }
        } else {
            let total = shape.reduce(1, *)
            N = total / 6
            get = { i, j in readMultiArrayValue(arr, linearIndex: i * 6 + j) }
        }

        var out: [YOLODetection] = []
        out.reserveCapacity(N)

        for i in 0..<N {
            var a = get(i, 0), b = get(i, 1), c = get(i, 2), d = get(i, 3)
            let score = get(i, 4)
            let cls = Int(get(i, 5).rounded())

            let maxCoord = max(max(abs(a), abs(b)), max(abs(c), abs(d)))
            let isNormalized = maxCoord <= 2.0
            if isNormalized {
                a *= inputW; c *= inputW
                b *= inputH; d *= inputH
            }

            if c <= a || d <= b {
                let cx = a, cy = b, w = c, h = d
                a = cx - w / 2.0
                b = cy - h / 2.0
                c = cx + w / 2.0
                d = cy + h / 2.0
            }

            let x1 = min(a, c), x2 = max(a, c)
            let y1 = min(b, d), y2 = max(b, d)

            out.append(YOLODetection(x1: x1, y1: y1, x2: x2, y2: y2, score: score, cls: cls))
        }

        return out
    }

    // MARK: - Box origin conversion

    /// Convert a detection to TOP-LEFT origin (internal canonical form).
    public static func detectionToTopLeft(
        _ d: YOLODetection,
        origin: BoxOrigin,
        inputH: Double
    ) -> YOLODetection {
        guard origin == .bottomLeft else { return d }
        let y1 = inputH - d.y2
        let y2 = inputH - d.y1
        return YOLODetection(x1: d.x1, y1: y1, x2: d.x2, y2: y2, score: d.score, cls: d.cls)
    }

    // MARK: - Private helpers

    private static func readMultiArrayValue(_ arr: MLMultiArray, linearIndex: Int) -> Double {
        switch arr.dataType {
        case .double:
            return arr.dataPointer.assumingMemoryBound(to: Double.self)[linearIndex]
        case .float32:
            return Double(arr.dataPointer.assumingMemoryBound(to: Float.self)[linearIndex])
        case .float16:
            let u = arr.dataPointer.assumingMemoryBound(to: UInt16.self)[linearIndex]
            let sign = (u & 0x8000) != 0
            let exp = Int((u & 0x7C00) >> 10)
            let frac = Int(u & 0x03FF)
            var f: Float
            if exp == 0 {
                f = Float(frac) / Float(1 << 10) * powf(2, -14)
            } else if exp == 0x1F {
                f = frac == 0 ? .infinity : .nan
            } else {
                f = (1.0 + Float(frac) / Float(1 << 10)) * powf(2, Float(exp - 15))
            }
            return Double(sign ? -f : f)
        default:
            return Double(truncating: arr[linearIndex] as NSNumber)
        }
    }
}
