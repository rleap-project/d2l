import logging
import os
import shutil
import subprocess


def count_file_lines(filename):  # Might be a bit faster with a call to "wc -l"
    i = 0
    with open(filename) as f:
        for i, _ in enumerate(f, 1):
            pass
    return i


def remove_duplicate_lines(filename):
    """ Removes in-place any duplicate line in the file. Will also reorder the lines as a side-effect """
    subprocess.call(['sort', '-u', '-o', filename, filename])


def read_file(filename):
    """ Read a file, line by line, ignoring end-of-line characters"""
    with open(filename) as f:
        for line in f:
            yield line.rstrip('\n')


def execute(command, **kwargs):
    stdout = open(kwargs["stdout"], 'w') if "stdout" in kwargs else None
    stderr = open(kwargs["stderr"], 'w') if "stderr" in kwargs else None
    cwd = kwargs["cwd"] if "cwd" in kwargs else os.getcwd()

    logging.info(f'Executing "{" ".join(map(str, command))}" on directory "{cwd}"')
    if stdout:
        logging.info(f'Standard output redirected to "{stdout.name}"')
    if stderr:
        logging.info(f'Standard error redirected to "{stderr.name}"')

    retcode = subprocess.call(command, cwd=cwd, stdout=stdout, stderr=stderr)

    if stdout:
        stdout.close()

    if stderr:
        stderr.close()

    if stderr is not None and os.path.getsize(stderr.name) == 0:  # Delete error log if empty
        os.remove(stderr.name)

    return retcode


def create_experiment_workspace(dirname, rm_if_existed=False):
    if rm_if_existed and os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)
