/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.tinkerpop.gremlin.process.computer.neuralnetwork;

import org.apache.tinkerpop.gremlin.LoadGraphWith;
import org.apache.tinkerpop.gremlin.process.AbstractGremlinProcessTest;
import org.apache.tinkerpop.gremlin.process.computer.ComputerResult;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Graph;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.structure.VertexProperty;
import org.junit.Test;

import java.util.Iterator;

import static org.apache.tinkerpop.gremlin.LoadGraphWith.GraphData.PPI;
import static org.junit.Assert.assertTrue;

public class GraphAttentionNetworkVertexProgramTest extends AbstractGremlinProcessTest {
    @Test
    @LoadGraphWith(PPI)
    public void shouldExecuteGAT() throws Exception {
        final ComputerResult result = graph.compute(graphProvider.getGraphComputer(graph).getClass()).
            program(GraphAttentionNetworkVertexProgram.build().create(graph)).submit().get();

        Graph resultGraph = result.graph();
        GraphTraversalSource g = resultGraph.traversal();

        // calculate micro-F1 score
        int truePositiveCount = 0, falsePositiveCount = 0, falseNegativeCount = 0;
        Iterator<Vertex> iter = g.V();
        while (iter.hasNext()) {
            Vertex v = iter.next();
            Iterator<VertexProperty<Number>> predIter = v.properties("pred");
            int i = 0;
            while (predIter.hasNext()) {
                Number pred = predIter.next().value();
                Number label = ((Number) v.property("label" + i++).value()).intValue();
                if (pred.equals(1) && label.equals(1)) {
                    truePositiveCount++;
                } else if (!pred.equals(label)) {
                    if (pred.equals(1)) {
                        falsePositiveCount++;
                    } else {
                        falseNegativeCount++;
                    }
                }
            }
        }
        double microF1Score = truePositiveCount / (truePositiveCount + 0.5 * (falsePositiveCount + falseNegativeCount));
        assertTrue(Math.abs(microF1Score - 0.9247) < 1e-3);
    }
}
