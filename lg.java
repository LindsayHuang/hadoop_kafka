import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class lg {

    public static class LinearRegressionMapper extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, FloatWritable>
    {

        private Path[] localFiles;

        FileInputStream fis = null;
        BufferedInputStream bis = null;


        @Override
        public void configure(JobConf job)
        {
            /**
             * Read the distributed cache
             */

            try {
                localFiles = DistributedCache.getLocalCacheFiles(job);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

        int counter = 0;

        public void map(LongWritable key, Text value, OutputCollector<LongWritable, FloatWritable> output,
                        Reporter reporter) throws IOException {

            String line = value.toString();

            String[] features = line.split("\\s+");

            List<Float> values = new ArrayList<Float>();

            /**
             * read the values and convert them to floats
             */
            for(int i = 0; i<features.length; i++)
            {
                values.add(new Float(features[i]));
            }

            /**
             * calculate the cost.
             */
            output.collect(new LongWritable(1), new FloatWritable(costs(values)));
        }

        private final float costs(List<Float> values)
        {
            /**
             * Load the cache files
             */

            File file = new File(localFiles[0].toString());

            float costs = 0;

            try {
                fis = new FileInputStream(file);
                bis = new BufferedInputStream(fis);

                BufferedReader d = new BufferedReader(new InputStreamReader(bis));
                String line = d.readLine();

                SimpleRegression r = new SimpleRegression(true);
                double [][] input = new double[values.size()][2];
                for(int j = 0; j < values.size(); j++)
                {
                    input[j][0] = j;
                    input[j][1] = values.get(j);

                }
                r.addData(input);
                costs = values.get(0)+(float)(r.getSlope() * (values.size()));

            } catch (Exception e) {
                e.printStackTrace();
            }



            return costs;

        }

    }




    public static class LinearRegressionReducer extends MapReduceBase implements Reducer<LongWritable, FloatWritable, LongWritable, FloatWritable>
    {


        public void reduce(LongWritable key, Iterator<FloatWritable> value,
                           OutputCollector<LongWritable, FloatWritable> output, Reporter reporter)
                throws IOException {


            /**
             * The reducer just has to sum all the values for a given key
             *
             */

            float sum = 0;

            while(value.hasNext())
            {
                sum += value.next().get();
            }

            output.collect(key, new FloatWritable(sum));

        }

    }


    /**
     * @param args
     */
    public static void main(String[] args) {

        //the class is LinearRegression
        JobConf conf = new JobConf(lg.class);

        //the jobname is linearregression (this can be anything)
        conf.setJobName("linearregression");

        /**
         * Try to load the theta values into the distributed cache
         */
        try {
            //make sure this is your path to the cache file in the hadoop file system
            DistributedCache.addCacheFile(
                    new URI(args[2]), conf);
        } catch (URISyntaxException e1) {
            e1.printStackTrace();
        }

        //set the output key class
        conf.setOutputKeyClass(LongWritable.class);
        //set the output value class
        conf.setOutputValueClass(FloatWritable.class);

        //set the mapper
        conf.setMapperClass(LinearRegressionMapper.class);
        //set the combiner
        conf.setCombinerClass(LinearRegressionReducer.class);
        //set the reducer
        conf.setReducerClass(LinearRegressionReducer.class);

        //set the input format
        conf.setInputFormat(TextInputFormat.class);
        //set the output format
        conf.setOutputFormat(TextOutputFormat.class);

        //set the input path (from args)
        FileInputFormat.setInputPaths(conf, new Path(args[0]));
        //set the output path (from args)
        FileOutputFormat.setOutputPath(conf, new Path(args[1]));

        //try to run the job
        try {
            JobClient.runJob(conf);
        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}
