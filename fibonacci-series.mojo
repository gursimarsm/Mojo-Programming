from Benchmark import Benchmark

fn fib(n: Int) -> Int:
    if n < 2: 
        return n
    else:
        return fib(n-1) + fib(n-2)


fn foo() -> None:
    let total = 35
    var summ = 0
    for i in range(total):
        summ += fib(i)

    let ans = (F32(summ)/total)
    print(ans)

def foo_test():
    fn test_fn():
        _ = foo()

    let secs = F64(Benchmark().run[test_fn]())
    print(secs)

foo_test()
