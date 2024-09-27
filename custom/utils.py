from pathlib import Path

def getOutputPath(input: Path, suffix: str, prefix: str):
    
    return input.parent.joinpath(prefix + input.stem + suffix)