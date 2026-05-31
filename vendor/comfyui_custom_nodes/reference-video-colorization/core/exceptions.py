"""Custom exceptions for ColorMNet integration."""


class ColorMNetError(Exception):
    """Base exception for all ColorMNet errors."""
    pass


class ModelNotFoundError(ColorMNetError):
    """Raised when model checkpoint is not found."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__(
            f"Model checkpoint not found at: {model_path}\n"
            f"Please run install.py to download the model, or check the path."
        )


class InsufficientVRAMError(ColorMNetError):
    """Raised when there's not enough VRAM for the operation."""
    def __init__(self, required_mb: int, available_mb: int):
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient VRAM: Required {required_mb}MB, but only {available_mb}MB available.\n"
            f"Try: 1) Lower resolution, 2) Enable FP16, 3) Reduce batch size, or 4) Use CPU mode"
        )


class InvalidInputError(ColorMNetError):
    """Raised when input validation fails."""
    pass


class ValidationError(ColorMNetError):
    """Raised when input validation fails."""
    pass
