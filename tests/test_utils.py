#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
import gzip
import io
import urllib.parse

import pytest

import smart_open.utils


class _ContextManager:
    def __init__(self, exit_result=None, exit_error=None, close_on_exit=None):
        self.exit_result = exit_result
        self.exit_error = exit_error
        self.close_on_exit = close_on_exit
        self.exit_args = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_args.append((exc_type, exc_value, traceback))
        if self.exit_error is not None:
            raise self.exit_error
        if self.close_on_exit is not None:
            self.close_on_exit.closed = True
        self.closed = True
        return self.exit_result


class _UploadingWriter(io.BytesIO):
    def __init__(self, upload_error):
        super().__init__()
        self.upload_error = upload_error
        self.close_calls = 0
        self.exit_args = []

    def close(self):
        """Simulate a writer whose upload fails when it closes."""
        self.close_calls += 1
        if self.upload_error is None:
            return super().close()
        if self.close_calls > 1:
            msg = "writer is already closed"
            raise ValueError(msg)
        raise self.upload_error

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_args.append((exc_type, exc_value, traceback))
        self.close()


class _ExitCountingBytesIO(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.exit_args = []

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_args.append((exc_type, exc_value, traceback))
        return super().__exit__(exc_type, exc_value, traceback)


@pytest.mark.parametrize(
    ("value", "minval", "maxval", "expected"),
    [
        (5, 0, 10, 5),
        (11, 0, 10, 10),
        (-1, 0, 10, 0),
        (10, 0, None, 10),
        (-10, 0, None, 0),
    ],
)
def test_clamp(value, minval, maxval, expected):
    """Clamp."""
    assert smart_open.utils.clamp(value, minval=minval, maxval=maxval) == expected


@pytest.mark.parametrize(
    ("value", "params", "expected"),
    [
        (10, {}, 10),
        (-10, {}, 0),
        (-10, {"minval": -5}, -5),
        (10, {"maxval": 5}, 5),
    ],
)
def test_clamp_defaults(value, params, expected):
    """Clamp defaults."""
    assert smart_open.utils.clamp(value, **params) == expected


def test_check_kwargs():
    """Check kwargs."""
    import smart_open.s3

    kallable = smart_open.s3.open
    kwargs = {"client": "foo", "unsupported": "bar", "client_kwargs": "boaz"}
    supported = smart_open.utils.check_kwargs(kallable, kwargs)
    assert supported == {"client": "foo", "client_kwargs": "boaz"}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("s3://bucket/key", ("s3", "bucket", "/key", "", "")),
        ("s3://bucket/key?", ("s3", "bucket", "/key?", "", "")),
        ("s3://bucket/???", ("s3", "bucket", "/???", "", "")),
        ("https://host/path?foo=bar", ("https", "host", "/path", "foo=bar", "")),
    ],
)
def test_safe_urlsplit(url, expected):
    """Safe urlsplit."""
    actual = smart_open.utils.safe_urlsplit(url)
    assert actual == urllib.parse.SplitResult(*expected)


def test_file_like_proxy_skips_inner_exit_when_outer_exit_fails():
    """Do not call inner cleanup after outer exit fails."""
    outer_error = RuntimeError("upload failed")
    outer = _ContextManager(exit_error=outer_error)
    inner = _ContextManager(exit_error=ValueError("cleanup failed"))
    proxy = smart_open.utils.FileLikeProxy(outer, inner)

    with pytest.raises(RuntimeError, match="upload failed") as exc_info, proxy:
        pass

    assert exc_info.value is outer_error
    assert inner.exit_args == []


def test_file_like_proxy_preserves_upload_error_during_text_wrapper_close():
    """Preserve a writer upload failure raised while the text wrapper closes."""
    upload_error = ConnectionError("upload failed")
    writer = _UploadingWriter(upload_error)
    text = io.TextIOWrapper(writer, encoding="utf-8")
    proxy = smart_open.utils.FileLikeProxy(text, writer)

    with pytest.raises(ConnectionError) as exc_info, proxy:
        proxy.write("payload")

    assert exc_info.value is upload_error
    assert writer.close_calls == 1
    assert writer.exit_args == []

    writer.upload_error = None
    text.close()


def test_file_like_proxy_forwards_body_exception_and_outer_suppression():
    """Forward a body failure to inner and retain outer suppression."""
    outer = _ContextManager(exit_result=True)
    inner = _ContextManager()
    proxy = smart_open.utils.FileLikeProxy(outer, inner)
    body_error = RuntimeError("body failed")

    with proxy:
        raise body_error

    assert inner.exit_args[0][0] is RuntimeError
    assert inner.exit_args[0][1] is body_error
    assert inner.closed


def test_file_like_proxy_exits_inner_left_open_by_text_wrapper_on_body_error():
    """Forward body errors to an open writer after the text wrapper exits."""
    writer = _ExitCountingBytesIO()
    text = smart_open.utils.TextIOWrapper(writer, encoding="utf-8")
    proxy = smart_open.utils.FileLikeProxy(text, writer)
    body_error = RuntimeError("body failed")

    with pytest.raises(RuntimeError) as exc_info, proxy:
        raise body_error

    assert exc_info.value is body_error
    assert writer.exit_args[0][0] is RuntimeError
    assert writer.exit_args[0][1] is body_error
    assert writer.closed


def test_file_like_proxy_skips_inner_exit_after_text_wrapper_closes_it():
    """Skip inner exit after a successful text wrapper close."""
    writer = _ExitCountingBytesIO()
    text = io.TextIOWrapper(writer, encoding="utf-8")
    proxy = smart_open.utils.FileLikeProxy(text, writer)

    with proxy:
        proxy.write("payload")

    assert writer.closed
    assert writer.exit_args == []


def test_file_like_proxy_exits_writer_left_open_by_compressor():
    """Exit an open binary writer after the gzip wrapper closes."""
    writer = _ExitCountingBytesIO()
    compressed = gzip.GzipFile(fileobj=writer, mode="wb")
    proxy = smart_open.utils.FileLikeProxy(compressed, writer)

    with proxy:
        proxy.write(b"payload")

    assert compressed.closed
    assert writer.closed
    assert writer.exit_args == [(None, None, None)]


def test_file_like_proxy_exits_a_shared_stream_once():
    """Exit one time when binary outer and inner are the same stream."""
    stream = _ContextManager()
    proxy = smart_open.utils.FileLikeProxy(stream, stream)

    with proxy:
        pass

    assert len(stream.exit_args) == 1


def test_file_like_proxy_returns_outer_exit_result():
    """Return the outer exit result after successful inner cleanup."""
    exit_result = object()
    outer = _ContextManager(exit_result=exit_result)
    inner = _ContextManager()
    proxy = smart_open.utils.FileLikeProxy(outer, inner)

    result = proxy.__exit__(None, None, None)

    assert result is exit_result
    assert inner.exit_args == [(None, None, None)]
    assert inner.closed


def test_file_like_proxy_skips_inner_that_outer_closed():
    """Do not exit an inner stream closed by the outer wrapper."""
    inner = _ContextManager()
    outer = _ContextManager(close_on_exit=inner)
    proxy = smart_open.utils.FileLikeProxy(outer, inner)

    with proxy:
        pass

    assert inner.closed
    assert inner.exit_args == []
