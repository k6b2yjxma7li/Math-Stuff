# Timer class
import time
# defining Timer as data class;
from dataclasses import dataclass, field
# adding typing features;
from typing import Callable, ClassVar, Dict, Optional
# decorator from Timer;
from contextlib import ContextDecorator


class TimerError(Exception):
    """Timer class exception"""


@dataclass
class Timer(ContextDecorator):
    """Class/Context manager/Decorator for timing purposes"""
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    start_msg: str = ""
    pause_msg: str = "Paused at {:0.4f} seconds."
    stop_msg: str = "Elapsed time: {:0.4f} seconds."
    logger: Optional[Callable[[str], None]] = print
    __start_time: Optional[float] = field(default=None, init=False, repr=False)
    __current_time: Optional[float] = field(default=None, init=False,
                                            repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer do dict of timers"""
        # Add new named timers to dictionary of timers
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start timer"""
        if self.__start_time is not None:
            raise TimerError("Timer is already running. Use .stop() to stop or"
                             " .pause() to pause it.")

        if self.logger:
            if self.logger is print:
                self.logger(self.start_msg, end="\n"*bool(self.start_msg))
            else:
                self.logger(self.start_msg)

        if self.__current_time is not None:
            self.__start_time = self.__current_time
            self.__current_time = None
        else:
            self.__start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timer"""
        if self.__start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it.")

        elapsed_time = time.perf_counter() - self.__start_time

        self.__start_time = None

        if self.logger:
            if self.logger is print:
                self.logger(self.stop_msg.format(elapsed_time),
                            end="\n"*bool(self.stop_msg))
            else:
                self.logger(self.stop_msg.format(elapsed_time))

        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def pause(self) -> float:
        """Pause counter"""
        if self.__start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it.")

        self.__current_time = self.__start_time
        elapsed_time = time.perf_counter() - self.__start_time

        if self.logger:
            if self.logger is print:
                self.logger(self.pause_msg.format(elapsed_time),
                            end="\n"*bool(self.pause_msg))
            else:
                self.logger(self.pause_msg.format(elapsed_time))

        self.__start_time = None

        return elapsed_time

    def __enter__(self):
        """Start timer in context manager mode"""
        self.start()
        return self

    def __exit__(self, *args):
        """Stop timer in context manager mode"""
        self.stop()
