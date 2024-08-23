class SequenceMatcher {
    constructor(a, b) {
        this.a = a
        this.b = b
    }

    ratio() {
        const matches = this.getMatchingBlocks()
        let sum = matches.reduce((acc, match) => acc + match[2], 0)
        return (2.0 * sum) / (this.a.length + this.b.length)
    }

    getMatchingBlocks() {
        let la = this.a.length
        let lb = this.b.length
        let queue = [[0, la, 0, lb]]
        let matching_blocks = []

        while (queue.length > 0) {
            let [alo, ahi, blo, bhi] = queue.pop()
            let x = this.findLongestMatch(alo, ahi, blo, bhi)
            let i = x[0], j = x[1], k = x[2]

            if (k > 0) {
                matching_blocks.push(x)
                if (alo < i && blo < j) {
                    queue.push([alo, i, blo, j])
                }
                if (i + k < ahi && j + k < bhi) {
                    queue.push([i + k, ahi, j + k, bhi])
                }
            }
        }

        matching_blocks.sort((a, b) => a[0] - b[0])

        let i1 = 0, j1 = 0, k1 = 0
        let non_adjacent = []
        for (let [i2, j2, k2] of matching_blocks) {
            if (i1 + k1 === i2 && j1 + k1 === j2) {
                k1 += k2
            } else {
                if (k1) non_adjacent.push([i1, j1, k1])
                i1 = i2
                j1 = j2
                k1 = k2
            }
        }
        if (k1) non_adjacent.push([i1, j1, k1])

        non_adjacent.push([la, lb, 0])
        return non_adjacent
    }

    findLongestMatch(alo, ahi, blo, bhi) {
        let a = this.a
        let b = this.b
        let besti = alo
        let bestj = blo
        let bestsize = 0
        let j2len = {}

        for (let i = alo; i < ahi; i++) {
            let newj2len = {}
            for (let j = blo; j < bhi; j++) {
                if (a[i] === b[j]) {
                    let k = newj2len[j] = (j2len[j - 1] || 0) + 1
                    if (k > bestsize) {
                        besti = i - k + 1
                        bestj = j - k + 1
                        bestsize = k
                    }
                }
            }
            j2len = newj2len
        }

        return [besti, bestj, bestsize]
    }
}

function stringSimilarity(a, b) {
    return new SequenceMatcher(a, b).ratio()
}