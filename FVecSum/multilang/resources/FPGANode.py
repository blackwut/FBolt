import os
# import time
import numpy as np
import storm
from wurlitzer import pipes
import pyopencl as cl
from CLFPGA import *
from pyopencl import command_execution_status as cs
# Thread-Safe https://docs.python.org/3/library/collections.html#deque-objects
from collections import deque


class FBoltAsync(storm.Bolt):

    def __init__(self,
                 xclbin_filepath=None,
                 kernel_name=None,
                 buffer_descriptors=None,
                 degree=1,
                 emulator=False,
                 profile=False):

        self.emulator = emulator
        if self.emulator:
            os.environ['XCL_EMULATION_MODE'] = 'sw_emu'

        with pipes() as (out, err):
            self.xilinx = CLXilinxDevice(xclbin_filepath)
            self.kernel = cl.Kernel(self.xilinx.program, kernel_name)
            self.kernel_queue = cl.CommandQueue(self.xilinx.context)

            self.degree = degree
            self.kernel_events = deque()

            self.tuples = deque()
            self.buffer_descriptors = buffer_descriptors
            self.buffers = []
            self.read_buffers = {}
            for bd in buffer_descriptors:
                if bd.is_IN():
                    b = FWriteBuffers(self.xilinx.context,
                                      bd.size_in_bytes(),
                                      self.degree,
                                      profile)
                    self.buffers.append(b)
                elif bd.is_OUT():
                    b = FReadBuffers(self.xilinx.context,
                                     bd.size_in_bytes(),
                                     self.degree,
                                     profile)
                    self.buffers.append(b)
                    # contains references to not yet read buffers
                    self.read_buffers[bd.name] = deque()
                else:
                    self.buffers.append(None)

            scalar_dtypes = []
            for bd in self.buffer_descriptors:
                if bd.is_SCALAR():
                    scalar_dtypes.append(bd.dtype)
                else:
                    scalar_dtypes.append(None)
            self.kernel.set_scalar_arg_dtypes(scalar_dtypes)

            self.count = 0
            self.profile = profile
            if self.profile:
                self.profilingManager = CLProfilingManager()

    def pop_oldest_event(self):
        if len(self.kernel_events) == self.degree:
            return self.kernel_events.popleft()
        return None

    def emit(self, tup, anchors=[]):
        m = {"command": "emit"}
        m["anchors"] = [a.id for a in anchors]
        m["tuple"] = tup
        storm.sendMsgToParent(m)
        # return storm.readTaskIds()

    def reading_callback(self, status, event, count, data, bd):
        if status == cs.COMPLETE:
            self.read_buffers[bd.name].append(data)

            once = True
            for v in self.read_buffers.values():
                if len(v) == 0:
                    once = False
                    break

            if once:
                if self.profile:
                    self.profilingManager.end(count, event)

                results = []
                for v in self.read_buffers.values():
                    results.append(v.popleft())

                tup = self.tuples.popleft()
                output = self.prepare_emit(tup, results)
                try:
                    self.emit(output, anchors=[tup])
                    storm.ack(tup)
                except Exception:
                    storm.reportError(storm.traceback.format_exc())
                    storm.fail(tup)

    def process(self, tup):
        self.tuples.append(tup)
        kernel_args = self.prepare_compute(tup)

        oldest_kernel_event = self.pop_oldest_event()

        wait_events = []
        for i, b, bd, arg in zip(range(len(kernel_args)),
                                 self.buffers,
                                 self.buffer_descriptors,
                                 kernel_args):
            if bd.is_IN():
                b.next()
                wait_events.append(b.write(arg, oldest_kernel_event))
                self.kernel.set_arg(i, b.current())
            elif bd.is_OUT():
                b.next()
                self.kernel.set_arg(i, b.current())
                oldest_event = b.pop_oldest_event()
                if oldest_event:
                    wait_events.append(oldest_event)
            elif bd.is_SCALAR():
                self.kernel.set_arg(i, arg)
            else:
                raise RuntimeError("bd.btype is unknown or not implemented!")

        if self.profile:
            self.profilingManager.start(self.count, wait_events)

        gws = lws = (1, 1, 1)
        event = cl.enqueue_nd_range_kernel(self.kernel_queue,
                                           self.kernel,
                                           gws, lws,
                                           wait_for=wait_events)
        self.kernel_events.append(event)

        for b, bd in zip(self.buffers, self.buffer_descriptors):
            if bd.is_OUT():
                data = np.empty(bd.elem_nums, dtype=bd.dtype)
                evt = b.read(data, [event])
                evt.set_callback(cs.COMPLETE,
                                 lambda _status, _event=evt, _count=self.count, _data=data, _bd=bd:
                                 self.reading_callback(_status, _event, _count, _data, _bd))
        self.count += 1

    def dump_profiling(self, filename):
        if self.profile:
            self.profilingManager.dump_to_file(filename)

    # This function must return a list of the kernel's arguments
    # Put None if a kernel argument is FBufferType.OUT
    def prepare_compute(self, tup):
        pass

    # This function must return a list containing the output tuple
    def prepare_emit(self, tup, results):
        pass

    def finish(self):
        for b, bd in zip(self.buffers, self.buffer_descriptors):
            if bd.is_OUT():
                b.finish()
        self.kernel_queue.finish()


class FBoltSync(storm.Bolt):

    def __init__(self,
                 xclbin_filepath=None,
                 kernel_name=None,
                 buffer_descriptors=None,
                 emulator=False,
                 profile=False):

        self.emulator = emulator
        if self.emulator:
            os.environ['XCL_EMULATION_MODE'] = 'sw_emu'

        with pipes() as (out, err):
            self.xilinx = CLXilinxDevice(xclbin_filepath)
            self.kernel = cl.Kernel(self.xilinx.program, kernel_name)
            self.kernel_queue = cl.CommandQueue(self.xilinx.context)

            self.buffer_descriptors = buffer_descriptors
            self.buffers = []
            for bd in buffer_descriptors:
                if bd.is_IN():
                    b = FWriteBuffers(self.xilinx.context,
                                      bd.size_in_bytes(),
                                      1,
                                      profile)
                    self.buffers.append(b)
                elif bd.is_OUT():
                    b = FReadBuffers(self.xilinx.context,
                                     bd.size_in_bytes(),
                                     1,
                                     profile)
                    self.buffers.append(b)
                else:
                    self.buffers.append(None)

            scalar_dtypes = []
            for bd in self.buffer_descriptors:
                if bd.is_SCALAR():
                    scalar_dtypes.append(bd.dtype)
                else:
                    scalar_dtypes.append(None)
            self.kernel.set_scalar_arg_dtypes(scalar_dtypes)

            self.profile = profile
            if self.profile:
                self.profilingManager = CLProfilingManager()
                self.count = 0

    def process(self, tup):
        kernel_args = self.prepare_compute(tup)

        write_wait_events = []
        for i, b, bd, arg in zip(range(len(kernel_args)),
                                 self.buffers,
                                 self.buffer_descriptors,
                                 kernel_args):
            if bd.is_IN():
                b.next()
                write_wait_events.append(b.write(arg))
                self.kernel.set_arg(i, b.current())
            elif bd.is_OUT():
                b.next()
                self.kernel.set_arg(i, b.current())
            elif bd.is_SCALAR():
                self.kernel.set_arg(i, arg)
            else:
                raise RuntimeError("bd.btype is unknown or not implemented!")

        if self.profile:
            self.count += 1
            self.profilingManager.start(self.count, write_wait_events)

        gws = lws = (1, 1, 1)
        event = cl.enqueue_nd_range_kernel(self.kernel_queue,
                                           self.kernel,
                                           gws, lws,
                                           wait_for=write_wait_events)

        read_buffers = []
        read_wait_events = []
        for b, bd in zip(self.buffers, self.buffer_descriptors):
            if bd.is_OUT():
                data = np.empty(bd.elem_nums, dtype=bd.dtype)
                read_buffers.append(data)
                evt = b.read(data, [event])
                read_wait_events.append(evt)

        cl.wait_for_events(read_wait_events)

        if self.profile:
            max_evt = None
            max_evt_time = 0
            for evt in read_wait_events:
                if evt.profile.end > max_evt_time:
                    max_evt = evt
                    max_evt_time = evt.profile.end

            self.profilingManager.end(self.count, max_evt)

        output = self.prepare_emit(tup, read_buffers)
        try:
            storm.emit(output, anchors=[tup])
            storm.ack(tup)
        except Exception:
            storm.reportError(storm.traceback.format_exc())
            storm.fail(tup)

    def dump_profiling(self, filename):
        if self.profile:
            self.profilingManager.dump_to_file(filename)

    # This function must return a list of the kernel's arguments
    # Put None if a kernel argument is FBufferType.OUT
    def prepare_compute(self, tup):
        pass

    # This function must return a list containing the output tuple
    def prepare_emit(self, tup, results):
        pass

    def finish(self):
        for b, bd in zip(self.buffers, self.buffer_descriptors):
            if bd.is_OUT():
                b.finish()
        self.kernel_queue.finish()

# class FBoltAsync(storm.Bolt):

#     def __init__(self,
#                  xclbin_filepath=None,
#                  kernel_name=None,
#                  buffer_descriptors=None,
#                  emulator=False,
#                  profile=False):

#         self.emulator = emulator
#         if self.emulator:
#             os.environ['XCL_EMULATION_MODE'] = 'sw_emu'

#         with pipes() as (out, err):
#             self.xilinx = CLXilinxDevice(xclbin_filepath)
#             self.kernel = cl.Kernel(self.xilinx.program, kernel_name)
#             self.kernel_queue = cl.CommandQueue(self.xilinx.context)

#             self.tuples = deque()
#             self.buffer_descriptors = buffer_descriptors
#             self.buffers = []
#             self.read_buffers = {}
#             for bd in buffer_descriptors:
#                 if bd.is_IN():
#                     b = FWriteBuffers(self.xilinx.context,
#                                       bd.size_in_bytes(),
#                                       bd.degree,
#                                       profile)
#                     self.buffers.append(b)
#                 elif bd.is_OUT():
#                     b = FReadBuffers(self.xilinx.context,
#                                      bd.size_in_bytes(),
#                                      bd.degree,
#                                      profile)
#                     self.buffers.append(b)
#                     self.read_buffers[bd.name] = deque()
#                 else:
#                     self.buffers.append(None)

#             scalar_dtypes = []
#             for bd in self.buffer_descriptors:
#                 if bd.is_SCALAR():
#                     scalar_dtypes.append(bd.dtype)
#                 else:
#                     scalar_dtypes.append(None)
#             self.kernel.set_scalar_arg_dtypes(scalar_dtypes)

#             self.profile = profile
#             self.count = 0
#             if self.profile:
#                 self.profilingManager = CLProfilingManager()

#     def emit(self, tup, anchors=[]):
#         m = {"command": "emit"}
#         m["anchors"] = [a.id for a in anchors]
#         m["tuple"] = tup
#         storm.sendMsgToParent(m)
#         return storm.readTaskIds()

#     def reading_callback(self, status, event, count, data, bd):
#         if status == cs.COMPLETE:
#             self.read_buffers[bd.name].append(data)

#             once = True
#             for v in self.read_buffers.values():
#                 if len(v) == 0:
#                     once = False
#                     break

#             if once:
#                 if self.profile:
#                     self.profilingManager.end(count, event)

#                 results = []
#                 for v in self.read_buffers.values():
#                     results.append(v.popleft())

#                 tup = self.tuples.popleft()
#                 output = self.prepare_emit(tup, results)
#                 try:
#                     self.emit(output, anchors=[tup])
#                     storm.ack(tup)
#                 except Exception:
#                     storm.reportError(storm.traceback.format_exc())
#                     storm.fail(tup)

#     def process(self, tup):
#         self.tuples.append(tup)
#         kernel_args = self.prepare_compute(tup)

#         wait_events = []
#         for i, b, bd, arg in zip(range(len(kernel_args)),
#                                  self.buffers,
#                                  self.buffer_descriptors,
#                                  kernel_args):
#             if bd.is_IN():
#                 b.next()
#                 wait_events.append(b.write(arg))
#                 self.kernel.set_arg(i, b.current())
#             elif bd.is_OUT():
#                 b.next()
#                 self.kernel.set_arg(i, b.current())
#             elif bd.is_SCALAR():
#                 self.kernel.set_arg(i, arg)
#             else:
#                 raise RuntimeError("bd.btype is unknown or not implemented!")

#         if self.profile:
#             self.count += 1
#             self.profilingManager.start(self.count, wait_events)

#         for b, bd in zip(self.buffers, self.buffer_descriptors):
#             if bd.is_OUT():
#                 b.wait_oldest_read()

#         gws = lws = (1, 1, 1)
#         event = cl.enqueue_nd_range_kernel(self.kernel_queue,
#                                            self.kernel,
#                                            gws, lws,
#                                            wait_for=wait_events)

#         for b, bd in zip(self.buffers, self.buffer_descriptors):
#             if bd.is_OUT():
#                 data = np.empty(bd.elem_nums, dtype=bd.dtype)
#                 evt = b.read(data, [event])
#                 evt.set_callback(cs.COMPLETE,
#                                  lambda _status, _event=evt, _count=self.count, _data=data, _bd=bd:
#                                  self.reading_callback(_status, _event, _count, _data, _bd))

#     def dump_profiling(self, filename):
#         self.profilingManager.dump_to_file(filename)

#     # This function must return a list of the kernel's arguments
#     # Put None if a kernel argument is FBufferType.OUT
#     def prepare_compute(self, tup):
#         pass

#     # This function must return a list containing the output tuple
#     def prepare_emit(self, tup, results):
#         pass


# class FBoltSync(storm.Bolt):

#     def __init__(self,
#                  xclbin_filepath=None,
#                  kernel_name=None,
#                  buffer_descriptors=None,
#                  emulator=False,
#                  profile=False):

#         self.emulator = emulator
#         if self.emulator:
#             os.environ['XCL_EMULATION_MODE'] = 'sw_emu'

#         with pipes() as (out, err):
#             self.xilinx = CLXilinxDevice(xclbin_filepath)
#             self.kernel = cl.Kernel(self.xilinx.program, kernel_name)
#             self.kernel_queue = cl.CommandQueue(self.xilinx.context)

#             self.buffer_descriptors = buffer_descriptors
#             self.buffers = []
#             for bd in buffer_descriptors:
#                 if bd.is_IN():
#                     b = FWriteBuffers(self.xilinx.context,
#                                       bd.size_in_bytes(),
#                                       1,
#                                       profile)
#                     self.buffers.append(b)
#                 elif bd.is_OUT():
#                     b = FReadBuffers(self.xilinx.context,
#                                      bd.size_in_bytes(),
#                                      1,
#                                      profile)
#                     self.buffers.append(b)
#                 else:
#                     self.buffers.append(None)

#             scalar_dtypes = []
#             for bd in self.buffer_descriptors:
#                 if bd.is_SCALAR():
#                     scalar_dtypes.append(bd.dtype)
#                 else:
#                     scalar_dtypes.append(None)
#             self.kernel.set_scalar_arg_dtypes(scalar_dtypes)

#             self.profile = profile
#             self.count = 0
#             if self.profile:
#                 self.profilingManager = CLProfilingManager()

#     def process(self, tup):
#         kernel_args = self.prepare_compute(tup)

#         write_wait_events = []
#         for i, b, bd, arg in zip(range(len(kernel_args)),
#                                  self.buffers,
#                                  self.buffer_descriptors,
#                                  kernel_args):
#             if bd.is_IN():
#                 b.next()
#                 write_wait_events.append(b.write(arg))
#                 self.kernel.set_arg(i, b.current())
#             elif bd.is_OUT():
#                 b.next()
#                 self.kernel.set_arg(i, b.current())
#             elif bd.is_SCALAR():
#                 self.kernel.set_arg(i, arg)
#             else:
#                 raise RuntimeError("bd.btype is unknown or not implemented!")

#         if self.profile:
#             self.count += 1
#             self.profilingManager.start(self.count, write_wait_events)

#         gws = lws = (1, 1, 1)
#         event = cl.enqueue_nd_range_kernel(self.kernel_queue,
#                                            self.kernel,
#                                            gws, lws,
#                                            wait_for=write_wait_events)

#         read_buffers = []
#         read_wait_events = []
#         for b, bd in zip(self.buffers, self.buffer_descriptors):
#             if bd.is_OUT():
#                 data = np.empty(bd.elem_nums, dtype=bd.dtype)
#                 read_buffers.append(data)
#                 evt = b.read(data, [event])
#                 read_wait_events.append(evt)

#         cl.wait_for_events(read_wait_events)

#         max_evt = None
#         max_evt_time = 0
#         for evt in read_wait_events:
#             if evt.profile.end > max_evt_time:
#                 max_evt = evt
#                 max_evt_time = evt.profile.end

#         if self.profile:
#             self.profilingManager.end(self.count, max_evt)

#         output = self.prepare_emit(tup, read_buffers)
#         try:
#             storm.emit(output, anchors=[tup])
#             storm.ack(tup)
#         except Exception:
#             storm.reportError(storm.traceback.format_exc())
#             storm.fail(tup)

#     def dump_profiling(self, filename):
#         if self.profile:
#             self.profilingManager.dump_to_file(filename)

#     # This function must return a list of the kernel's arguments
#     # Put None if a kernel argument is FBufferType.OUT
#     def prepare_compute(self, tup):
#         pass

#     # This function must return a list containing the output tuple
#     def prepare_emit(self, tup, results):
#         pass
