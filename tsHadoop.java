import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;
import java.io.*;
import java.util.List;
import weka.core.Instances;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.WekaForecaster;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class tsHadoop {

    public static class TimeSeriesMapper extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, FloatWritable>
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

            try {
                // load the wine data
                String line = value.toString();
                String res = line.replace('?','\n');
                Reader inputString = new StringReader(res);
                Instances wine = new Instances(new BufferedReader(inputString));
                // new forecaster
                WekaForecaster forecaster = new WekaForecaster();

                // set the targets we want to forecast. This method calls
                forecaster.setFieldsToForecast("num");
                forecaster.setBaseForecaster(new GaussianProcesses());

                forecaster.getTSLagMaker().setTimeStampField("time"); // date time stamp
                forecaster.getTSLagMaker().setMinLag(1);
                forecaster.getTSLagMaker().setMaxLag(12); // monthly data

                // add a month of the year indicator field
                forecaster.getTSLagMaker().setAddMonthOfYear(true);

                // add a quarter of the year indicator field
                forecaster.getTSLagMaker().setAddQuarterOfYear(true);

                // build the model
                forecaster.buildForecaster(wine, System.out);

                // prime the forecaster with enough recent historical data
                // to cover up to the maximum lag. In our case, we could just supply
                // the 12 most recent historical instances, as this covers our maximum
                // lag period
                forecaster.primeForecaster(wine);

                // forecast for 12 units (months) beyond the end of the
                // training data
                List<List<NumericPrediction>> forecast = forecaster.forecast(12, System.out);

                // output the predictions. Outer list is over the steps; inner list is over
                // the targets
                for (int i = 0; i < 12; i++) {
                    List<NumericPrediction> predsAtStep = forecast.get(i);
                    //System.out.println(predsAtStep.toString());
                    NumericPrediction predForTarget = predsAtStep.get(0);
                    output.collect(new LongWritable(i+1), new FloatWritable(((float)predForTarget.predicted())));
                }

            } catch (Exception ex) {
                ex.printStackTrace();
            }

        }

    }




    public static class TimeSeriesReducer extends MapReduceBase implements Reducer<LongWritable, FloatWritable, LongWritable, FloatWritable>
    {


        public void reduce(LongWritable key, Iterator<FloatWritable> value,
                           OutputCollector<LongWritable, FloatWritable> output, Reporter reporter)
                throws IOException {


            /**
             * The reducer just has to sum all the values for a given key
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
        JobConf conf = new JobConf(tsHadoop.class);

        //the jobname is linearregression (this can be anything)
        conf.setJobName("TimeSeries");

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
        conf.setMapperClass(TimeSeriesMapper.class);
        //set the combiner
        conf.setCombinerClass(TimeSeriesReducer.class);
        //set the reducer
        conf.setReducerClass(TimeSeriesReducer.class);

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




