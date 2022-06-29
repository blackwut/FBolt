package FVecSum;

import java.util.Map;
import org.apache.storm.task.ShellBolt;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.tuple.Fields;

public class FBolt extends ShellBolt implements IRichBolt {

    public FBolt() {
        super("/usr/bin/python3", "fbolt.py");
        // this.changeChildCWD(false);
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("result", "timestamp"));
    }

    @Override
    public Map<String, Object> getComponentConfiguration() {
        return null;
    }
}
