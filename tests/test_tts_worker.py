"""Scheduling regressions, not substitute TTS/LLM acceptance measurements."""
import asyncio
import unittest

from tools.tts_server import _await_worker


class WorkerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_waits_for_actual_executor_worker(self):
        import threading

        started = threading.Event()
        finish = threading.Event()

        def worker():
            started.set()
            if not finish.wait(3):
                raise TimeoutError('Test did not release worker')
            return 'completed'

        future = asyncio.get_running_loop().run_in_executor(None, worker)
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        task = asyncio.create_task(_await_worker(future))
        try:
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done(), 'Cannot release model lock while thread runs')
            self.assertFalse(future.cancelled())
        finally:
            finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(future.result(), 'completed')

    async def test_worker_error_remains_visible(self):
        future = asyncio.get_running_loop().create_future()
        future.set_exception(ValueError('load failed'))
        with self.assertRaises(ValueError) as raised:
            await _await_worker(future)
        self.assertEqual(str(raised.exception), 'load failed')

    async def test_error_after_cancel_does_not_replace_cancellation(self):
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(_await_worker(future))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        future.set_exception(ValueError('load failed after stop'))
        with self.assertLogs('airi_tts', level='ERROR'):
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsInstance(future.exception(), ValueError)


if __name__ == '__main__':
    unittest.main()
