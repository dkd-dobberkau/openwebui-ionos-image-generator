from pydantic import BaseModel, Field
import requests
import json
import os
from datetime import datetime


class Pipe:
    """
    IONOS Image Generator Pipe with Inline Image Display

    This pipe generates images via IONOS AI Model Hub and displays the image
    directly in the chat response using markdown.
    """

    class Valves(BaseModel):
        # API Configuration
        IONOS_API_KEY: str = Field(
            default="", description="Your IONOS API Key (required for image generation)"
        )
        MODEL: str = Field(
            default="stabilityai/stable-diffusion-xl-base-1.0",
            description="Image generation model to use",
        )
        IMAGE_SIZE: str = Field(
            default="1024x1024", description="Size of generated images (width x height)"
        )
        SAVE_TO_DISK: bool = Field(
            default=False, description="Whether to save generated images to disk"
        )

    def __init__(self):
        # Initialize valves configuration
        self.valves = self.Valves()

    def pipe(self, body: dict, __user__: dict = None):
        """
        Generate image and return it as a markdown image in the response
        """
        # Get API token from valves or environment
        api_token = self.valves.IONOS_API_KEY or os.getenv("IONOS_API_TOKEN")

        if not api_token:
            return self._format_response(
                "Error: IONOS API key is not configured. Please set the IONOS_API_KEY valve or the IONOS_API_TOKEN environment variable."
            )

        # Extract user's prompt from the last message
        user_prompt = ""
        if "messages" in body and len(body["messages"]) > 0:
            for message in reversed(body["messages"]):
                if message.get("role") == "user":
                    user_prompt = message.get("content", "")
                    break

        if not user_prompt:
            return self._format_response(
                "Please provide a description of the image you want to generate."
            )

        # Make API request
        try:
            endpoint = "https://openai.inference.de-txl.ionos.com/v1/images/generations"

            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }

            request_body = {
                "model": self.valves.MODEL,
                "prompt": user_prompt,
                "size": self.valves.IMAGE_SIZE,
                "response_format": "b64_json",  # Ensure we get base64
            }

            # Log the request details (without API key)
            request_log = {
                "endpoint": endpoint,
                "model": self.valves.MODEL,
                "prompt": user_prompt,
                "size": self.valves.IMAGE_SIZE,
            }

            print(f"Making IONOS API request: {json.dumps(request_log)}")

            response = requests.post(endpoint, json=request_body, headers=headers)

            # Check if request was successful
            if response.status_code != 200:
                error_message = f"API request failed with status {response.status_code}: {response.text}"
                print(error_message)
                return self._format_response(error_message)

            # Get response data
            response_data = response.json()

            # Check if response contains image data
            if (
                "data" in response_data
                and len(response_data["data"]) > 0
                and "b64_json" in response_data["data"][0]
            ):
                # Get the base64 data
                base64_data = response_data["data"][0]["b64_json"]

                # Format the response with the image
                result = f"""
# Image Generated!

## Prompt:
"{user_prompt}"

## Model:
{self.valves.MODEL}

## Generated Image:
![Generated Image](data:image/jpeg;base64,{base64_data})

*Generated via IONOS AI Model Hub*
"""

                return self._format_response(result)
            else:
                # If no image data in response
                error_message = (
                    f"No image data found in API response: {json.dumps(response_data)}"
                )
                print(error_message)
                return self._format_response(error_message)

        except Exception as e:
            error_message = f"Error generating image: {str(e)}"
            print(error_message)
            return self._format_response(error_message)

    def _format_response(self, message: str) -> dict:
        """Format a response in OpenAI-compatible format"""
        return {
            "id": "ionos-response",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "ionos-image-generator",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
        }
