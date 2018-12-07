import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
logfile = logging.FileHandler('logfile')
stdfile = logging.StreamHandler()

logfile.setLevel(logging.WARNING)
stdfile.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logfile)
logger.addHandler(stdfile)

logger.debug(' DDebug ')
logger.info(" IInfo ")
logger.warning(' WWarning ')
logger.error(' EError ')
logger.critical(' CCritical ')

###  https://my.oschina.net/leejun2005/blog/126713
###  https://www.cnblogs.com/nancyzhu/p/8551506.html 

