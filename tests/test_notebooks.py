import os
from pathlib import Path

import pytest

pathname = Path("../examples")
files = pathname.glob("*ipynb")

testdir = pathname / "build"

if pathname.is_dir():
    pathname.rmdir()
pathname.mkdir()


@pytest.mark.notebooks
@pytest.mark.parametrize("file", files)
def test_notebook(file) -> None:

    cwd = os.getcwd()

    os.chdir(pathname)

    try:
        # run autotest on each notebook
        cmd = (
            "jupyter "
            + "nbconvert "
            + "--ExecutePreprocessor.timeout=600 "
            + "--to "
            + "notebook "
            + "--execute "
            + '"{}" '.format(file)
            + "--output-dir "
            + "{} ".format(testdir)
        )
        ival = os.system(cmd)
        msg = "could not run {}".format(file)
        assert ival == 0, msg
        assert os.path.isfile(os.path.join(testdir, file)), msg
    except Exception as e:
        os.chdir(cwd)
        raise Exception(e)
    os.chdir(cwd)
