import Foundation

extension String {
    func leadingNaturalNumber() -> String {
        let pattern = #"^\s*(\d+)"#
        guard
            let re = try? NSRegularExpression(pattern: pattern),
            let m = re.firstMatch(in: self, range: NSRange(self.startIndex..., in: self)),
            let r = Range(m.range(at: 1), in: self),
            let number = Int(self[r])
        else { return "" }
        return "\(number)"
    }
}
