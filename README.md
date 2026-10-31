# microsoft/Florence-2-base Cog Model

This is an implementation of [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:

    cog predict -i image=@car.jpg -i prompt="<CAPTION>"

# Input

![car](car.jpg)

# Output

    {'<CAPTION>': 'A green car parked in front of a yellow building.'}

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) is released under the [MIT](https://huggingface.co/microsoft/Florence-2-base/resolve/main/LICENSE) license.
