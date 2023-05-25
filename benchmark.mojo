from Benchmark import Benchmark

fn bench() -> Int:
    fn _iter():
        for i in range(10000):
            pass

    return Benchmark().run[_iter]()

print(bench())
