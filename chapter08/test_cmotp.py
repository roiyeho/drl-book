from cmotp.cmotp import CMOTP
from cmotp.env_config import CMOTPConfig
import time

env = CMOTP(CMOTPConfig())

env.reset()
env.render()

time.sleep(5)









