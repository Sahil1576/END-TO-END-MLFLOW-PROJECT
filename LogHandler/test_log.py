from logger import logger

def do_something():
    print("Doing Something right now..")


logger.info("Start some task..")
do_something()
logger.info("End task..")