// MARK: - BusTrackingFlow

/// Thin adapter: wraps BusTracker, conforming to the Flow protocol.
/// Input: [BusDetection]. Output: [TrackedBus].
public final class BusTrackingFlow: Flow {

    public let id: String
    private let tracker: BusTracker

    public init(id: String = "BusTracking", tracker: BusTracker) {
        self.id = id
        self.tracker = tracker
    }

    public func run(input: [BusDetection]) async throws -> [TrackedBus] {
        tracker.update(detections: input)
    }
}
