# Third-Party Notices

## Microsoft Swin Transformer

- **Project:** Swin Transformer
- **URL:** https://github.com/microsoft/Swin-Transformer
- **Local reference commit:** f82860bfb5225915aca09c3227159ee9e1df874d
- **Copyright:** Copyright (c) Microsoft Corporation.

The file `models/swin3d_aniso.py` adapts design patterns from the official
Swin Transformer V1 implementation (`models/swin_transformer.py`) and
rewrites them as a dynamic-size 3D anisotropic version for volumetric
medical image registration. No code is directly copied; the adaptation
follows the same architectural blueprint (window partition, shifted windows,
relative position bias, patch merging).

### MIT License

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
