## Chess Board Detection on PDFs

This script takes in a pdf files and tries to recognize chess diagrams on it. It then tries to
convert the chess board images to text representations. The result is stored in a similarly named
HTML file.

### Installation

First you need to install [Python][1]. Then you should install the `cbdetect.pyz` file from the
release page of this repository. Just put the pyz file in your home directory.

### Running

Now you need to open powershell. Make sure that you are located in the same folder as cbdetect.pyz
which you downloaded in the previous step. Now you can run the program by typing:

```
python cbdetect.pyz <FILE.pdf>
```

So if you for instance has a pdf file called `foo.pdf` in the same directory you just type:

```
python cbdetect.pyz foo.pdf
```

You may also omitt the pdf file argument in which case the program will launch a familiar file open
dialog. Note however that this might sometimes not work very well together with a screen reader like
NVDA.

After the pdf file has been selected properly, the program searches through all pages for chess
diagrams. The result will be written to an HTML file with the same name as the pdf file. Each page
in the pdf will have a corresponding HTML heading and all possible chess diagrams will be printed at
the top of each page.


[1]: https://www.python.org/downloads/
