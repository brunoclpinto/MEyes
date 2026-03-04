// MARK: - WorkflowOutput

/// Collected results from a workflow run, keyed by flow ID.
public struct WorkflowOutput: Sendable {
    private var store: [String: Any] = [:]

    public init() {}

    public mutating func set<T: Sendable>(_ value: T, for flowId: String) {
        store[flowId] = value
    }

    /// Type-safe retrieval. Returns nil if the key is missing or the type doesn't match.
    public func get<T: Sendable>(_ type: T.Type, for flowId: String) -> T? {
        store[flowId] as? T
    }
}
