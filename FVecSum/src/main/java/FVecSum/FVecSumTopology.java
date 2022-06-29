package FVecSum;

import org.apache.storm.topology.ConfigurableTopology;
import org.apache.storm.topology.TopologyBuilder;

public class FVecSumTopology extends ConfigurableTopology {

    final static String topologyName = "FVecSum";

    public static void main(String[] args) throws Exception {
        ConfigurableTopology.start(new FVecSumTopology(), args);
    }

    @Override
    protected int run(String[] args) throws Exception {

        TopologyBuilder builder = new TopologyBuilder();
        builder.setSpout("spout", new RandomVecSpout(), 1);
        builder.setBolt("fbolt", new FBolt(), 1).shuffleGrouping("spout");
        builder.setBolt("check", new CheckBolt(), 1).shuffleGrouping("fbolt");

        // conf.setDebug(true);
        conf.setNumWorkers(3);
        // conf.setTopologyWorkerMaxHeapSize(2048);

        conf.registerSerialization(org.apache.storm.shade.org.json.simple.JSONArray.class);

        return submit(topologyName, conf, builder);
    }
}
