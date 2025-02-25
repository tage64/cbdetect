## Chess Board Detection on PDFs

This script takes in a PDF file and tries to recognize chess diagrams on every page. It then tries
to convert the chess board images to text representations. The result is stored in a similarly named
HTML file.

### Installation

First you need to install [Python][1]. Then download the `cbdetect.pyz` file from [the release page
of this repository][2]. Just put the PYZ file in your home directory or somewhere else where it is
easy to find.

### Running

Now you need to open powershell. Make sure that you are located in the same folder as `cbdetect.pyz`
that you downloaded in the previous step. Now you can run the program by typing:

```
python cbdetect.pyz <FILE.pdf>
```

So if you for instance has a pdf file called `foo.pdf` in the same directory you just type:

```
python cbdetect.pyz foo.pdf
```

You may also omitt the pdf file argument in which case the program will launch a familiar file open
dialog. If you are using a screen reader, it might take a few seconds before this dialog appears, and
you might need to press Alt+Tab a couple of times to find it.

After the pdf file has been selected properly, the program searches through all pages for chess
diagrams. The result will be written to an HTML file with the same name as the pdf file. Each page
in the pdf will have a corresponding HTML heading and all possible chess diagrams will be printed at
the top of each page.


[1]: https://www.python.org/downloads/
[2]: https://github.com/tage64/cbdetect/releases/latest
