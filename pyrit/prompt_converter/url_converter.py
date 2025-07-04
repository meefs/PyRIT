# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import urllib.parse

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class UrlConverter(PromptConverter):
    """
    Converts a prompt to a URL-encoded string.
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into a URL-encoded string.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=urllib.parse.quote(prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
