alias _CLOCK_REALTIME = 0

@value
@register_passable("trivial")
struct _CTimeSpec:
    var tv_sec: Int  # Seconds
    var tv_nsec: Int  # NanoSeconds

    fn __init__() -> Self:
        return Self {tv_sec: 0, tv_nsec: 0}

    fn as_nanoseconds(self) -> Int:
        return self.tv_sec * 1_000_000_000 + self.tv_nsec


@always_inline
fn _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()
    let ts_pointer = Pointer[_CTimeSpec].address_of(ts)

    let clockid_si32 = __mlir_op.`pop.cast`[
        _type : __mlir_type.`!pop.scalar<si32>`,
    ](clockid.value)

    # Call libc's clock_gettime.
    __mlir_op.`pop.external_call`[
        func : "clock_gettime".value,
        _type:None,
    ](clockid_si32, ts_pointer.address)

    return ts

## new code starts here

@value
@register_passable("trivial")
struct C_tm:
    var tm_sec: SI32
    var tm_min: SI32
    var tm_hour: SI32
    var tm_mday: SI32
    var tm_mon: SI32
    var tm_year: SI32
    var tm_wday: SI32
    var tm_yday: SI32
    var tm_isdst: SI32
    
    fn __init__() -> Self:
        return Self {
            tm_sec: 0,
            tm_min: 0,
            tm_hour: 0,
            tm_mday: 0,
            tm_mon: 0,
            tm_year: 0,
            tm_wday: 0,
            tm_yday: 0,
            tm_isdst: 0
        }

@always_inline
fn _ts_to_tm(owned ts: _CTimeSpec) -> C_tm:
    let ts_pointer = Pointer[Int].address_of(ts.tv_sec)

    # Call libc's clock_gettime.
    let tm = __mlir_op.`pop.external_call`[
        func : "gmtime".value,
        _type:Pointer[C_tm],
    ](ts_pointer).load()


    return tm

@value
struct Instant:
    """Seconds since epoch"""
    var seconds: Int
    """Nanos since second"""
    var nanos: Int
    
    fn __init__(inout self):
        self.seconds = 0
        self.nanos = 0
        
    @staticmethod
    fn utc_now() -> Self:
        let ts = _clock_gettime(_CLOCK_REALTIME)
        return Instant(ts.tv_sec, ts.tv_nsec)

@value
struct DateTimeLocal:
    var second: SI32
    var minute: SI32
    var hour: SI32
    var day_of_month: SI32
    var month: SI32
    var year: SI32
    var day_of_week: SI32
    var day_of_year: SI32
    var is_daylight_savings: Bool
    
    @staticmethod
    fn from_instant(instant: Instant) -> Self:
        let ts = _CTimeSpec(instant.seconds, instant.nanos)
        let tm = _ts_to_tm(ts)
        
        return DateTimeLocal (
            tm.tm_sec,
            tm.tm_min,
            tm.tm_hour,
            tm.tm_mday,
            tm.tm_mon + 1,
            tm.tm_year + 1900,
            tm.tm_wday,
            tm.tm_yday,
            not tm.tm_isdst.__bool__()
        )
        
    fn __str__(self) -> StringLiteral:
        """Format using ISO 8601"""
        return "TODO: implement when string formatting exists"
        
# Tests

let now = Instant.utc_now()
let dt = DateTimeLocal.from_instant(now)
print("second: ", dt.second)
print("minute: ", dt.minute)
print("hour: ", dt.hour)
print("day_of_month: ", dt.day_of_month)
print("month: ", dt.month)
print("year: ", dt.year)
print("weekday: ", dt.day_of_week)
print("day_of_year: ", dt.day_of_year)
print("is_daylight_savings: ", dt.is_daylight_savings)
