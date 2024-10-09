# modules for multiprocessing
import enum
import traceback
import sys
import atexit
import os 
import time
from functools import partial as bind

class Parallel:
    '''
    multiprocessing class that runs a function in parallel.
    suitable for running classes with multiple functions.
    '''
    def __init__(self, ctor, strategy):
        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        '''
        Get attribute of the worker.
        if 
        '''
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)
        
    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        '''

        '''
        state = state or ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            result = callable(getattr(state, name))
        elif message == PMessage.CALL:
            result = getattr(state, name)(*args, **kwargs)
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy = "thread", state = False):
        # wrap function to adapt to ProcessPipelineWorker
        if not state:
            fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    
    def __call__(self, *args, **kwargs):
        '''
        Call the function with the given arguments and return the result.
        return: Future class, a callable object that returns the result of a computation when called.
        '''
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise
    
    def wait(self):
        '''
        Wait for the worker to finish.
        '''
        return self.impl.wait()
    
    def close(self):
        '''
        Close the worker.
        '''
        self.impl.close()



class ProcessPipeWorker:
    '''
    A class that creates a worker process that can run a function in parallel.
    '''
    def __init__(self, fn, initializers = (), daemon = False):
        import multiprocessing as mp
        import cloudpickle

        self.fn = fn
        self.initializers = initializers
        self.daemon = daemon
        self._context = mp.get_context("spawn")
        self._parent_conn, _child_conn = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(
            target = self._loop, args=(_child_conn, fn, initializers), daemon = self.daemon
            )
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        return self._submit(Message.RUN, (args, kwargs))
    
    def wait(self):
        pass

    def close(self):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.
        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    os.kill(self._process.pid, 9)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass
    
    def _submit(self, message, payload = None):
        # send data to worker
        callid = self._nextid
        self._nextid += 1
        self._parent_conn.send((message, callid, payload))
        # returned message is stored in dictionary and popped when called
        return Future(self._receive, callid)
    
    def _receive(self, callid):
        # receive result from worker
        # store result in dictionary and pop it when called
        while callid not in self._results:
            try: 
                message, callid, payload = self._parent_conn.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                    raise Exception(payload)
            assert message == Message.RESULT, message
            self._results[callid] = payload
        return self._results.pop(callid)
    
    @staticmethod
    def _loop(pipe, fn, initializers):
        try:
            callid = None
            state = None
            import cloudpickle

            # load function and initializers
            initializers = cloudpickle.loads(initializers)
            fn = cloudpickle.loads(fn)
            # run initializers
            [initializer() for initializer in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue
                message, callid, payload = pipe.recv()
                # tell main process that worker is ready
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                # stop worker
                elif message == Message.STOP:
                    return
                # run function
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = fn(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            return
        # catch exceptions and send them to main process
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return
        # close pipe if worker is done
        finally:
            try:
                pipe.close()
            except Exception:
                pass        

class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    '''
    A future is a callable object that returns the result of a computation when called.
    '''
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result
        
class Damy:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()