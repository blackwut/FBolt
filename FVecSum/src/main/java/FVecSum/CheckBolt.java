/*
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version
 * 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package FVecSum;

import java.io.IOException;
import java.util.Map;
import java.util.ArrayList;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;

// For logging
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class CheckBolt extends BaseBasicBolt {

    private static final Logger logger = LogManager.getLogger(CheckBolt.class);

    private Sampler latency;
    private static final long samplingRate = 0; // adds every element
    private long count;


    @Override
    public void prepare(Map<String,Object> topoConf, TopologyContext context) {
        latency = new Sampler(samplingRate);
        count = 0;
    }

    @Override
    public void execute(Tuple tuple, BasicOutputCollector collector) {
        ArrayList<Long> result = (ArrayList<Long>)tuple.getValue(0);
        long timestamp = (long)tuple.getValue(1);

        boolean first = true;
        boolean success = true;
        for (Long r : result) {
            if (first == true) {
                first = false;
            } else {
                if (r != 1) {
                    success = false;
                    break;
                }
            }
        }

        if (!success) {
            logger.error("This result is WRONG!!!");
        }

        long now = System.nanoTime();
        latency.add((double)(now - timestamp) / 1000000.0, now);
        this.count += 1;
    }

    @Override
    public void cleanup() {
        try {
            MetricGroup.add("latency", latency);
            MetricGroup.dumpAll();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        // declarer.declare(new Fields("word", "count"));
    }
}
