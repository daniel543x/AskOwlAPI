"""Incremental decoding of one string value from streamed tool-call args.

Providers stream tool-call arguments as fragments of a JSON object, e.g.
`{"an`, `swer": "Rust`, ` is..."}`. ToolArgStream tolerates arbitrary
fragment boundaries (including inside escape sequences and unicode
escapes) and emits the decoded value incrementally. Shared by every
pipeline that streams a tool-call argument to an SSE consumer.
"""


_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


class ToolArgStream:
    """Incrementally decodes one string value from streamed tool-call args.

    Providers stream tool-call arguments as fragments of a JSON object, e.g.
    `{"an`, `swer": "Rust`, ` is..."}`. This state machine tolerates
    arbitrary fragment boundaries (including inside escape sequences and
    unicode escapes) and emits the decoded value incrementally.
    """

    def __init__(self, key: str):
        self._tokens = ('"', key, '"', ':', '"')
        self._token_index = 0
        self._in_token = False
        self._token_pos = 0
        self._state = "match"  # match | value | escape | unicode | done
        self._unicode_digits = ""
        self._decoded: list[str] = []
        self._emitted = 0
        self._pending = ""
        self.saw_value = False

    def feed(self, fragment: str) -> str:
        """Consume a fragment; return newly decoded text."""
        self._pending += fragment
        self._process()
        return self._emit()

    def finish(self) -> str:
        """Mark the stream complete; flush any decoded-but-unemitted text."""
        if self._state in ("value", "escape", "unicode"):
            self._state = "done"
        return self._emit()

    # <----- internals ----->

    def _emit(self) -> str:
        new_text = "".join(self._decoded[self._emitted :])
        self._emitted = len(self._decoded)
        return new_text

    def _process(self) -> None:
        while self._pending and self._state != "done":
            char = self._pending[0]
            self._pending = self._pending[1:]

            if self._state == "match":
                self._match(char)
            elif self._state == "value":
                self._value(char)
            elif self._state == "escape":
                self._escape(char)
            elif self._state == "unicode":
                self._unicode(char)

    def _match(self, char: str) -> None:
        token = self._tokens[self._token_index]

        if not self._in_token:
            # Whitespace between tokens is skipped.
            if char.isspace():
                return
            if char != token[0]:
                # Not the expected token start; restart matching.
                self._token_index = 0
                return
            self._in_token = True
            self._token_pos = 1
            if self._token_pos == len(token):
                self._advance_token()
            return

        # Inside a token: characters must match exactly.
        if self._token_pos < len(token) and char == token[self._token_pos]:
            self._token_pos += 1
            if self._token_pos == len(token):
                self._advance_token()
        else:
            # Mismatch: restart from the first token.
            self._token_index = 0
            self._in_token = False

    def _advance_token(self) -> None:
        self._in_token = False
        if self._token_index == len(self._tokens) - 1:
            # Consumed the opening quote of the value: value starts here.
            self._state = "value"
            self.saw_value = True
        else:
            self._token_index += 1

    def _value(self, char: str) -> None:
        if char == "\\":
            self._state = "escape"
        elif char == '"':
            self._state = "done"
        else:
            self._decoded.append(char)

    def _escape(self, char: str) -> None:
        if char == "u":
            self._state = "unicode"
            self._unicode_digits = ""
        else:
            self._decoded.append(_JSON_ESCAPES.get(char, char))
            self._state = "value"

    def _unicode(self, char: str) -> None:
        self._unicode_digits += char
        if len(self._unicode_digits) == 4:
            try:
                self._decoded.append(chr(int(self._unicode_digits, 16)))
            except ValueError:
                self._decoded.append(self._unicode_digits)
            self._state = "value"
