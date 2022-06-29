package FVecSum;

import java.util.Map;
import java.util.ArrayList;
import java.util.Random;
import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;

public class RandomVecSpout extends BaseRichSpout {

    final static int max_size = 8 * 1024;

    SpoutOutputCollector collector;
    Random rand;
    Integer count;

    @Override
    public void open(Map<String, Object> conf, TopologyContext context, SpoutOutputCollector collector) {
        this.collector = collector;
        rand = new Random();
        this.count = 0;
    }

    @Override
    public void nextTuple() {
        Utils.sleep(1);

        Integer size = rand.nextInt(max_size - 1) + 1;
        ArrayList<Integer> A = new ArrayList<Integer>(size);
        ArrayList<Integer> B = new ArrayList<Integer>(size);

        A.add(this.count);
        B.add(0);
        this.count++;

        for (int i = 1; i < size; ++i) {
            Integer value = rand.nextInt(max_size);
            A.add(value + 1);
            B.add(-value);
        }

        long timestamp = System.nanoTime();
        collector.emit(new Values(A, B, timestamp));
    }

    @Override
    public void ack(Object id) {
    }

    @Override
    public void fail(Object id) {
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("A", "B", "timestamp"));
    }
}
