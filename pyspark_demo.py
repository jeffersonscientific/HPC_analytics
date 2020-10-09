import pyspark
import time
#
fname = 'data/serc_usage_20200914.out'
#conf = pyspark.SparkConf().setAppName("read text file in pyspark").set("spark.cores.max", "2")
conf = pyspark.SparkConf('local[*]').set("spark.cores.max", "6").set("spark.executor.instances", "4").set("spark.executor.cores","1").set("spark.executor.memory", "14g").set("spark.executor.pyspark.memory", "13g")
#conf = conf.set("spark.executor.memory", "14g").set("spark.executor.pyspark.memory", "13g")
sc = pyspark.SparkContext(conf=conf)
#
lines = sc.textFile(fname)
delim = '|'
#
rows = lines.map(lambda s:s.split(delim))
#
print('** lines: ')
for ln in lines.take(10):
	print('** **: ', ln)
#
print('rows: ')
for rw in rows.take(10):
	print('** **: ', rw)
#
#
print('** rows again: ')
for rw in rows.take(10):
        print('** **: ', rw)
#
print('begin all_rows[]: ')
t0 = time.time()
#
all_rows = rows.collect()
#
delta_t = time.time()-t0
print('** time: {}'.format(delta_t))
#
with open('output_file.out', 'w') as fout:
    fout.write("time: time.time()-t0)\n")
    for rw in all_rows:
        fout.write('{}\n'.format(','.join(all_rows) ) )


#sc.close()

