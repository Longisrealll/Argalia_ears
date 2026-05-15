import asyncio
import time

RUUNOR = True
RUNN = True

async def main():
    global RUUNOR
    global RUNN
    while True:
        print("true")
        if(RUUNOR):
            asyncio.create_task(counter())
        if(RUNN):
            asyncio.create_task(secsis5())
        await asyncio.sleep(1)

async def counter():
    global RUUNOR
    RUUNOR = False
    for i in range(3):
        await asyncio.sleep(1)
    print("not so much")
    RUUNOR = True

async def secsis5():
    global RUNN
    RUNN = False
    for i in range(5):
        await asyncio.sleep(1)
    print("not so much now eh")
    RUNN = True

asyncio.run(main())
