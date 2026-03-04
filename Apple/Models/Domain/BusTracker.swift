// Pure tracking logic — zero image framework imports.

// MARK: - TrackedBus

public struct TrackedBus: Sendable {
    public let id: Int
    public let name: String          // "Bus1", "Bus2", ...
    public let lastBoxDetector: Box  // TOP-LEFT origin, detector space
    public let lastScore: Double
    public let approachingScore: Double
    public let isApproaching: Bool

    public init(id: Int, name: String, lastBoxDetector: Box,
                lastScore: Double, approachingScore: Double, isApproaching: Bool) {
        self.id = id; self.name = name; self.lastBoxDetector = lastBoxDetector
        self.lastScore = lastScore; self.approachingScore = approachingScore
        self.isApproaching = isApproaching
    }
}

// MARK: - BusTracker

/// Stateful IoU-based tracker. Zero CoreImage/Vision/CoreML imports — pure logic only.
public final class BusTracker {

    public struct Config: Sendable {
        public var iouMatchThreshold: Double = 0.2
        public var maxMissedFrames: Int = 40
        public var approachWindow: Int = 5
        public var approachMinFrames: Int = 2
        public var approachRatioThreshold: Double = 0.001
        public init() {}
    }

    public var config: Config

    // MARK: Private state

    private struct Track {
        let numericId: Int
        let name: String
        var lastBoxDetector: Box
        var lastScore: Double
        var lastSeenFrame: Int
        var ageFrames: Int
        var areaHistory: [Double]
        var approachingScore: Double
        var isApproaching: Bool
    }

    private var nextId: Int = 1
    private var tracks: [Int: Track] = [:]
    private var frameIndex: Int = 0

    public init(config: Config = .init()) {
        self.config = config
    }

    // MARK: - Public API

    /// Update tracks with new detections. Returns currently tracked (and approaching) buses.
    public func update(detections: [BusDetection]) -> [TrackedBus] {
        frameIndex += 1

        let matchedIds = matchAndUpdateTracks(detections: detections)
        pruneStaleTracks()

        return matchedIds.sorted().compactMap { id -> TrackedBus? in
            guard let t = tracks[id] else { return nil }
            return TrackedBus(
                id: t.numericId, name: t.name,
                lastBoxDetector: t.lastBoxDetector, lastScore: t.lastScore,
                approachingScore: t.approachingScore, isApproaching: t.isApproaching
            )
        }
    }

    // MARK: - Matching

    private func matchAndUpdateTracks(detections: [BusDetection]) -> [Int] {
        let detBoxes = detections.map { $0.boxDetector }

        var pairs: [(iou: Double, tid: Int, di: Int)] = []
        pairs.reserveCapacity(tracks.count * max(1, detBoxes.count))

        for (tid, tr) in tracks {
            for (di, db) in detBoxes.enumerated() {
                let v = iou(tr.lastBoxDetector, db)
                if v >= config.iouMatchThreshold { pairs.append((v, tid, di)) }
            }
        }
        pairs.sort { $0.iou > $1.iou }

        var usedTracks = Set<Int>()
        var detAssigned = Array(repeating: false, count: detBoxes.count)
        var matchedIds: [Int] = []

        for p in pairs {
            if usedTracks.contains(p.tid) || detAssigned[p.di] { continue }
            usedTracks.insert(p.tid)
            detAssigned[p.di] = true
            matchedIds.append(p.tid)
            updateTrack(id: p.tid, box: detBoxes[p.di], score: detections[p.di].score)
        }

        for (di, assigned) in detAssigned.enumerated() where !assigned {
            let id = nextId; nextId += 1
            let box = detBoxes[di]
            var tr = Track(
                numericId: id, name: "Bus\(id)",
                lastBoxDetector: box, lastScore: detections[di].score,
                lastSeenFrame: frameIndex, ageFrames: 1,
                areaHistory: [box.area], approachingScore: 0, isApproaching: false
            )
            computeApproach(&tr)
            tracks[id] = tr
            matchedIds.append(id)
        }

        return matchedIds
    }

    private func updateTrack(id: Int, box: Box, score: Double) {
        guard var tr = tracks[id] else { return }
        tr.lastBoxDetector = box
        tr.lastScore = score
        tr.lastSeenFrame = frameIndex
        tr.ageFrames += 1

        tr.areaHistory.append(box.area)
        if tr.areaHistory.count > config.approachWindow {
            tr.areaHistory.removeFirst(tr.areaHistory.count - config.approachWindow)
        }

        computeApproach(&tr)
        tracks[id] = tr
    }

    private func computeApproach(_ tr: inout Track) {
        let h = tr.areaHistory
        guard h.count >= config.approachMinFrames else {
            tr.approachingScore = 0; tr.isApproaching = false; return
        }

        let first = max(h.first ?? 0, 1e-6)
        let last  = max(h.last  ?? 0, 0)
        let ratio = (last / first) - 1.0
        tr.approachingScore = ratio

        var increases = 0
        for i in 1..<h.count { if h[i] >= h[i - 1] { increases += 1 } }
        let mostlyIncreasing = increases >= max(1, (h.count - 1) * 2 / 3)

        tr.isApproaching = (ratio >= config.approachRatioThreshold) && mostlyIncreasing
    }

    private func pruneStaleTracks() {
        let cutoff = frameIndex - config.maxMissedFrames
        tracks = tracks.filter { $0.value.lastSeenFrame >= cutoff }
    }

    // MARK: - IoU

    private func iou(_ a: Box, _ b: Box) -> Double {
        let xA = max(a.x1, b.x1), yA = max(a.y1, b.y1)
        let xB = min(a.x2, b.x2), yB = min(a.y2, b.y2)
        let inter = max(0, xB - xA) * max(0, yB - yA)
        let union = a.area + b.area - inter
        if union <= 0 { return 0 }
        return inter / union
    }
}
