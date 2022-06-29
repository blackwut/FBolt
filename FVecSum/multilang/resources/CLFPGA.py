import pyopencl as cl
from pyopencl import mem_flags as mf
from pyopencl import command_queue_properties as cqp
from collections import deque
from enum import Enum


class CLEventProfile:
    def __init__(self, name):
        self.name = name

    def start(self, events_start):
        self.events_start = events_start

    def end(self, event_end):
        self.event_end = event_end

    def start_ms(self):
        return 1e-6 * min([s.profile.start for s in self.events_start])

    def stop_ms(self):
        return 1e-6 * self.event_end.profile.end

    def time_ms(self):
        assert(self.events_start)
        assert(self.event_end)

        return 1e-6 * max([self.event_end.profile.end - s.profile.start
                           for s in self.events_start])

    def to_string(self):
        return "{0}\t{1:.8f}\t{2:.8f}\t{3:.8f}\n".format(self.name,
                                                         self.start_ms(),
                                                         self.stop_ms(),
                                                         self.time_ms())


class CLProfilingManager:

    def __init__(self):
        self.runs = {}
        self.completed = []

    def start(self, name, events):
        r = CLEventProfile(name)
        r.start(events)
        self.runs[name] = r

    def end(self, name, event):
        self.runs[name].end(event)
        self.completed.append(self.runs[name])
        del self.runs[name]

    def dump_to_file(self, filename):
        with open(filename, 'a') as f:
            for r in self.completed:
                f.write(r.to_string())


class CLXilinxDevice:

    def __init__(self, xclbin_filepath):
        # get the platform
        platforms = cl.get_platforms()
        found = False
        for p in platforms:
            platform_name = p.get_info(cl.platform_info.NAME)
            if platform_name == "Xilinx":
                self.platform = p
                found = True
                break

        if not found:
            raise RuntimeError("`Xilinx` Platform not Found!")

        # get the fpga device
        devices = self.platform.get_devices(cl.device_type.ACCELERATOR)
        if len(devices) == 0:
            raise RuntimeError("Device not found!")

        self.device = devices[0]

        # create a context
        self.context = cl.Context([self.device])

        # load the bitstream
        self.binary = open(xclbin_filepath, "rb").read()
        self.program = cl.Program(self.context, [self.device], [self.binary])
        self.program.build()


class FBuffers:

    def __init__(self, context, flags, size, degree, profile=False):
        self.context = context
        self.flags = flags
        self.size = size
        self.idx = -1
        self.degree = degree

        if profile:
            self.queue = cl.CommandQueue(self.context,
                                         properties=cqp.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(self.context)

        self.events = deque()
        self.buffers = []
        for i in range(self.degree):
            self.buffers.append(cl.Buffer(self.context, self.flags, size))

    def current(self):
        return self.buffers[self.idx % self.degree]

    def next(self):
        self.idx += 1
        return self.current()

    def pop_oldest_event(self):
        if len(self.events) == self.degree:
            return self.events.popleft()
        return None

    def finish(self):
        self.queue.finish()


class FWriteBuffers(FBuffers):

    def __init__(self, context, size, degree, profile=False):
        super().__init__(context,
                         mf.HOST_WRITE_ONLY | mf.READ_ONLY,
                         size,
                         degree,
                         profile)

    def write(self, src, wait_for=None, is_blocking=False):
        wait_events = []
        oldest_event = self.pop_oldest_event()
        if oldest_event:
            wait_events.append(oldest_event)
        if wait_for:
            wait_events.append(wait_for)

        event = cl.enqueue_copy(self.queue,
                                self.current(),
                                src,
                                wait_for=wait_events,
                                is_blocking=is_blocking)
        self.events.append(event)
        return event


class FReadBuffers(FBuffers):

    def __init__(self, context, size, degree, profile=False):
        super().__init__(context,
                         mf.HOST_READ_ONLY | mf.WRITE_ONLY,
                         size,
                         degree,
                         profile)

    def read(self, dest, wait_for=None, is_blocking=False):
        event = cl.enqueue_copy(self.queue,
                                dest,
                                self.current(),
                                wait_for=wait_for,
                                is_blocking=is_blocking)
        self.events.append(event)
        return event


class FBufferType(Enum):
    IN = 1
    OUT = 2
    SCALAR = 3


class FBufferDescriptor:
    def __init__(self,
                 btype,
                 dtype,
                 name,
                 elem_nums):
        self.btype = btype
        self.dtype = dtype
        self.name = name
        self.elem_size = self.dtype(1).nbytes
        self.elem_nums = elem_nums

    def size_in_bytes(self):
        return self.elem_size * self.elem_nums

    def is_IN(self):
        return (self.btype is FBufferType.IN)

    def is_OUT(self):
        return (self.btype is FBufferType.OUT)

    def is_SCALAR(self):
        return (self.btype is FBufferType.SCALAR)
