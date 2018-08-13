/*
 *  # coding: utf-8
 *  # Copyright (C) 2017.  UKP lab
 *  #
 *  # Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
 *  # Licensed under the MIT license
 */

var TheApp = {};

TheApp.init =  function() {
    $("#getanswerbutton").click(function(){
        $(".resultsrow").find(".panel").hide();
        TheApp.interface.before_query();

        var question_text = $("#questionField").val();
        TheApp.request.current_request.question = question_text;
        TheApp.request.current_request.simplified_npparser = 1;
        /* Clear the last response */
        TheApp.response.last_response = $.extend({}, TheApp.request.current_request);

        TheApp.request.post_question_text(TheApp.request.current_request)
            .then(function(response) {
                TheApp.interface.process_ug(response);
                return TheApp.request.post("/question-answering/groundedgraphs/", response);
            })
            .then(function(response) {
                TheApp.interface.process_grounded(response);
                return TheApp.request.post("/question-answering/evaluategraphs/", response.graphs.slice(0,1));
            })
            .then(function(response) {
                TheApp.interface.process_results(response);
                return TheApp.request.post("/question-answering/encoderweigths/", {gs: TheApp.response.last_response.gs, ug: TheApp.response.last_response.ug});
            })
            .then(TheApp.interface.process_encoder_weights)
            .done(function(){TheApp.interface.after_query();});
    });

    $("#getsemanticvectorsbutton").click(function() {
        TheApp.interface.before_any_query();
        TheApp.request.post("/question-answering/semvectors/", {
            gs: TheApp.response.last_response.gs,
            ug: TheApp.response.last_response.ug
        })
            .then(function(response) {
                TheApp.interface.process_sem_vectors(response);
                TheApp.interface.after_any_query();
            })
    });

    TheApp.draw.vvis = d3.select("#vectorsField svg");
    TheApp.draw.vvis = TheApp.draw.vvis.append("g").attr("transform", "translate(" + TheApp.draw.vvismargin.left + "," + TheApp.draw.vvismargin.top + ")")
    ;

    TheApp.draw.cvis = d3.select("#cnnWeightsField svg");
    TheApp.draw.cvis = TheApp.draw.cvis.append("g").attr("transform", "translate(" + TheApp.draw.cvismargin.left + "," + TheApp.draw.cvismargin.top + ")");

    $('[data-toggle="tooltip"]').tooltip();

    $("#report-wrong").click(function () {
        TheApp.request.report({question: TheApp.response.last_response.question, answers: TheApp.response.last_response.answers})
            .then(function(){
                $("#report-wrong").tooltip('hide').fadeOut(function () { var that = $(this).siblings(".glyphicon-ok").fadeIn().tooltip('show');
                                                                        setTimeout(function(){ that.tooltip('hide');}, 1000);})
            });
    });
};

/* Requests */

TheApp.request = {};
TheApp.response = {};

TheApp.request.default_num_entity_options = 1;
TheApp.request.current_request = {};

TheApp.response.t0 = 0;
TheApp.response.t1 = 0;
TheApp.response.last_response = {};

TheApp.request.post = function(url, postdata){
    return $.ajax({
        url: url,
        type: 'POST',
        error: function(){
            $("#answerRow").find(".panel-danger").show();
            TheApp.interface.after_query();
        },
        complete: function(data){
        },
        data: JSON.stringify(postdata),
        dataType: "json",
        processData: false,
        cache: false,
        contentType: "application/json"
    });
};

TheApp.request.post_question_text = function(question_text) {
    return TheApp.request.post("/question-answering/ungroundedgraph/", question_text);
};

TheApp.request.report = function(postdata){
    $("#answerRow").find(".panel-danger").hide();
    return $.ajax({
        url: "/question-answering/reporterror/",
        type: 'POST',
        error: function(){
            $("#answerRow").find(".panel-danger").show();
        },
        success: function (){
            console.log("Thank you!")
        },
        data: JSON.stringify(postdata),
        processData: false,
        cache: false,
        contentType: "application/json"
    });
};


/* Updating interface */

TheApp.interface = {};

TheApp.interface.before_any_query = function(){
    $(".btn").attr("disabled", "disabled");
    $("#linksField").find(".list-group-item").addClass("disabled");
};

TheApp.interface.before_query = function(){
    TheApp.interface.before_any_query();
    var answer_row = $("#answerRow");
    answer_row.find(".panel").hide();
    answer_row.find("#report-wrong").show();
    answer_row.find(".glyphicon-ok").hide();
    var progress_panel = $("#progressRow");
    progress_panel.find(".progress-bar").width("10%");
    progress_panel.find(".progress").show();

    var sem_vectors_panel = $("#vectorsRow").find(".panel");
    sem_vectors_panel.find("#getsemanticvectorsbutton").show();
    TheApp.draw.vvis.selectAll("*").remove();

    $("#statisticsRow").find(".panel")
        .show()
        .find("#statsField").empty();
    TheApp.response.t0 = performance.now();
};

TheApp.interface.after_any_query = function(){
    $(".btn").removeAttr("disabled");
    $("#linksField").find(".list-group-item").removeClass("disabled");
};

TheApp.interface.after_query = function(){
    $("#progressRow").find(".progress-bar").width("100%")
        .parent().hide();
    $(".btn").removeAttr("disabled");
    TheApp.response.t1 = performance.now();
    $("#statsField").append($("<span />", {'text': "Processing time: " + ((TheApp.response.t1 - TheApp.response.t0) / 1000).toFixed(2) + " seconds"}));

    var sem_vectors_panel = $("#vectorsRow").find(".panel");
    sem_vectors_panel.show();
};


TheApp.interface.process_ug = function(ungrounded_graph) {
    TheApp.response.last_response.ug = ungrounded_graph;
    $("#progressRow").find(".progress-bar").width("50%");
    $("#taggedRow").find(".panel").show();
    $("#entitiesField").empty();
    $.each(ungrounded_graph.fragments, function(index, value) {
        $('<span />', {
            'text': value["tokens"].join(" ") + ":" + value["type"].trim()
        }).addClass("label label-primary inline-block").appendTo("#entitiesField");
    });

    $("#taggedField").empty();
    $.each(ungrounded_graph.tagged, function(index, value) {
        var tokentagged = $('<div />').addClass("token-inline").appendTo("#taggedField");
        $('<span />', {
            'text': value['word']
        }).addClass("text-inline").appendTo(tokentagged);
        $('<span />', {
            'text': value['pos']
        }).addClass("tag-inline").appendTo(tokentagged);
        if (value['ner'] !== "O") {
            $('<span />', {
                'text': value['ner']
            }).addClass("tag-ne tag-inline").appendTo(tokentagged);
        }
    });


    $("#linksRow").find(".panel").show();
    $("#linksField").find(".entty").remove();
    $.each(ungrounded_graph.entities, function(eindex, entity) {
        var entty = $("#linksField").find("div.tmpl").clone();
        entty.find(".etitle > .elabel").text(entity.tokens.join(" "));
        var linkings = entity.linkings.slice();

        $.each(entity.linkings, function(lindex, linking) {
            var opt = entty.find("a.tmpl").clone();
            opt.find(".elabel").text(linking[1]);
            opt.find(".qid").text(linking[0]);
            opt.attr('id', linking[0]);
            opt.removeClass("tmpl hidden").addClass("disabled").appendTo(entty.find("div.list-group"));
            if (lindex < TheApp.request.default_num_entity_options){
                opt.addClass("active");
            }
            opt.click(function(){
                entity.linkings = [];
                $(this).toggleClass("active");
                TheApp.interface.before_query();
                entty.find(".list-group-item").each(function(index){
                    var lnkng = $(this);
                    if (lnkng.hasClass("active")){
                        entity.linkings.push(linkings[index-1]);
                    }
                });
                TheApp.request.post("/question-answering/groundedgraphs/", ungrounded_graph)
                    .then(TheApp.interface.process_grounded)
                    .done(function(){TheApp.interface.after_query();})
                    .then(TheApp.interface.process_results);
            });
        });
        entty.removeClass("tmpl hidden").addClass("entty").appendTo("#linksField");
        entity.linkings = linkings.slice(0, TheApp.request.default_num_entity_options);
    });
};

TheApp.interface.process_grounded = function(response) {
    var grounded_graphs = response.graphs;
    TheApp.response.last_response.gs = grounded_graphs;

    $("#progressRow").find(".progress-bar").width("80%");
    $("#linksField").find(".entty a.list-group-item").removeClass("disabled");

    // Draw graphs
    $("#graphsRow").find(".panel").show();
    $("#graphsField").find(".grph").remove();
    var graphSvgTmpl = $("#graphsField").find("div.tmpl");
    for (var s_g = 0; s_g < 6 && s_g < grounded_graphs.length; s_g++){
        var graphSvg = graphSvgTmpl.clone().removeClass("tmpl hidden").addClass("grph").appendTo("#graphsField");
        graphSvg.find("h4 span").text((s_g+1) + ".");
        graphSvg.find("h4 small").text(grounded_graphs[s_g][1].toFixed(4));
        if (s_g === 0){
            graphSvg.find(".thumbnail").addClass("selected");
        }
        TheApp.draw.draw_graph(grounded_graphs[s_g][0], graphSvg);
    }
};
TheApp.interface.process_results = function(results){
    TheApp.response.last_response.answers = results;
    $("#answerRow").find(".panel-success").show();
    $("#answerfield").empty();
    $.each(results[1], function(index, value) {
        if (value !== null && value.length > 0) {
            $('<a />', {
                'text': value.trim(),
                'href': "https://tools.wmflabs.org/reasonator/?&q=" + results[0][index].substring(1),
                'target': "_blank",
                'title': "Go to a detailed answer page"
            }).addClass("btn btn-default answr").appendTo("#answerfield");
        }
    });
};

TheApp.interface.process_encoder_weights = function(response){
    // Draw graph cnn weights
    $("#cnnWeightsRow").find(".panel").show();
    TheApp.draw.drawCnnWeights(response.tokens,
        response.cnnweights);
};

TheApp.interface.process_sem_vectors = function(response){
    var vectors = response.vectors;

    // Draw graph vectors
    var sem_vectors_panel = $("#vectorsRow").find(".panel");
    sem_vectors_panel.show();
    sem_vectors_panel.find("#getsemanticvectorsbutton").hide();

    var data = [];
    if (vectors.length > 0) {
        data = [
            {x: vectors[0][0], y: vectors[0][1], name: 'Your question', class: "dot-question",  sc:1, dot: 4}
        ];
    }
    var relevant_relations = new Set();
    for (var j in TheApp.response.last_response.gs){
        var graph = TheApp.response.last_response.gs[j][0];
        for (var k in graph.edgeSet) {
            relevant_relations.add(graph.edgeSet[k].propertyName);
        }
    }

    for(var v_i = 1; v_i < vectors.length; v_i++){
        var vector = vectors[v_i];
        // var graph = grounded_graphs[v_i-1][0];
        // var graph_string = [];
        // for (var edge_i in graph.edgeSet){
        //     graph_string.push(graph.edgeSet[edge_i].propertyName)
        // }
        var relation_string = response.relations[v_i-1];
        if (relevant_relations.has(relation_string)){
            data.push({x: vector[0], y: vector[1], name: relation_string, class: "dot-graph-relevant", sc:8, dot: 3});
        } else {
            data.push({x: vector[0], y: vector[1], name: relation_string, class: "dot-graph", sc:8, dot: 2});
        }

    }
    TheApp.draw.drawVectors(data);
};

/* Drawing */
TheApp.draw = {};
TheApp.draw.vvis = {};
TheApp.draw.vvismargin = {top: 20, right: 20, bottom: 30, left: 40};
TheApp.draw.cvismargin = {top: 20, right: 20, bottom: 30, left: 80};

TheApp.draw.draw_graph = function(s_g, graphSvg){
    var gvis = d3.select(graphSvg.get(0)).select("svg");
    gvis.selectAll("*").remove();
    var nodes = [
        { x:   0, y: 0, name: "Q var", class: "qnode"}
    ];
    var links = [];
    for (var edge_i in s_g.edgeSet){
        var edge = s_g.edgeSet[edge_i];
        for (var n_i=0; n_i < nodes.length; n_i ++){
            links.push({ source: n_i, target: nodes.length, name: edge.kbID.slice(0,-1) + ":" + edge.propertyName })
        }
        nodes.push({ x:   0, y: 0, name: edge.rightkbID + ":" + edge.canonical_right, class: "node"});
    }

    var link = gvis.selectAll('.link')
        .data(links)
        .enter().append('line')
        .attr('class', function(d) { return ((d.source === 0) ? "link" : "link-dummy" ); });

    var width = graphSvg.find("svg").innerWidth(), height = graphSvg.find("svg").innerHeight();
    var force = d3.forceSimulation()
        .force("charge", d3.forceManyBody().strength(50))
        .force('centerX', d3.forceX(width / 3))
        .force('centerY', d3.forceY(height / 2));

    var node = gvis.selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('class', function(d) { return d.class; })
        .attr('r', 15)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    var text = gvis.selectAll(".node-label")
        .data(nodes).enter()
        .append("g")
        .attr('class', "node-label");
    text.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", function(d) { return (d.name.length + 1) + "ex"; })
        .attr("height", 22)
        .attr("rx", 10)
        .attr("rz", 10);
    text.append("text")
        .attr("x", 4)
        .attr("y", 16)
        .text(function(d) { return d.name; });

    var reltext = gvis.selectAll(".rel-label")
        .data(links).enter()
        .append("g")
        .filter(function(d) { return d.source === 0; })
        .attr('class', "rel-label");
    reltext.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", function(d) { return (d.name.length + 1) + "ex"; })
        .attr("height", 22)
        .attr("rx", 5)
        .attr("rz", 5);
    reltext.append("text")
        .attr("x", 4)
        .attr("y", 16)
        .text(function(d) { return d.name; });

    function tick() {
        link.attr('x1', function(d) { return d.source.x; })
            .attr('y1', function(d) { return d.source.y; })
            .attr('x2', function(d) { return d.target.x; })
            .attr('y2', function(d) { return d.target.y; });
        node.attr("transform", transform);
        text.attr("transform", transform);
        reltext.attr("transform", function(d){ return "translate(" + ((d.source.x + d.target.x)/2 -11) + "," + ((d.source.y + d.target.y)/2 -11) + ")"; } );
    }

    function transform(d) {
        return "translate(" + d.x + "," + d.y + ")";
    }

    force.nodes(nodes).on("tick", tick);
    force.force("link", d3.forceLink(links));
    force.force("link").distance(function(d) { return (d.source.index === 0) ? 150 : 150 ; });


    function dragstarted(d) {
        if (!d3.event.active) force.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    }

    function dragended(d) {
        if (!d3.event.active) force.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
};

TheApp.draw.drawVectors = function(data){
    TheApp.draw.vvis.selectAll("*").remove();
    var width = $("#vectorsField").find("svg").innerWidth(), height = $("#vectorsField").find("svg").innerHeight();
    width = width - TheApp.draw.vvismargin.left - TheApp.draw.vvismargin.right;
    height = height - TheApp.draw.vvismargin.top - TheApp.draw.vvismargin.bottom;

    var x = d3.scaleLinear()
        .range([0, width]);

    var y = d3.scaleLinear()
        .range([height, 0]);

    var sc_size = Math.sqrt(Math.pow(width,2) + Math.pow(height,2));
    var sc = d3.scaleLinear()
        .range([0, sc_size ]);

    var xAxis = d3.axisBottom(x);

    var yAxis = d3.axisLeft(y);

    var x0 = d3.extent(data, function(d) { return d.x; }),
        y0 = d3.extent(data, function(d) { return d.y; }),
        sc0 = [0, 1];
    x.domain(x0).nice();
    y.domain(y0).nice();
    sc.domain(sc0);

    TheApp.draw.vvis.append("g")
        .attr("class", "x axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    TheApp.draw.vvis.append("g")
        .attr("class", "y axis axis--y")
        .call(yAxis);

    TheApp.draw.vvis.selectAll("circle")
        .data(data)
        .enter().append("circle")
        .attr("class", function(d) { return d.class; })
        .attr("r",  function(d) { return d.dot; })
        .attr("cx", function(d) { return x(d.x); })
        .attr("cy", function(d) { return y(d.y); });

    var textcontainer = TheApp.draw.vvis.append("g");
    textcontainer.selectAll("text")
        .data(data)
        .enter().append("text")
        .attr("class", "scattertext")
        .attr("x", 0)
        .attr("y", 0)
        .attr("transform", function(d) { return "translate(" + (x(d.x) + 6) + "," + (y(d.y) - 2) + ") scale(" + sc(1) /sc_size + ")"; })
        .text(function(d) { return d.name; });

    var brush = d3.brush().on("end", brushended),
        idleTimeout,
        idleDelay = 350;

    TheApp.draw.vvis.append("g")
        .attr("class", "brush")
        .call(brush);

    function brushended() {
        var s = d3.event.selection;
        if (!s) {
            if (!idleTimeout) return idleTimeout = setTimeout(idled, idleDelay);
            x.domain(x0).nice();
            y.domain(y0).nice();
            sc.domain(sc0);
        } else {
            x.domain([s[0][0], s[1][0]].map(x.invert, x));
            y.domain([s[1][1], s[0][1]].map(y.invert, y));
            sc.domain([0,  Math.sqrt(Math.pow((s[0][0]-s[1][0]),2) + Math.pow(s[1][1]-s[0][1],2))].map(sc.invert, sc));
            TheApp.draw.vvis.select(".brush").call(brush.move, null);
        }
        zoom();
    }

    function idled() {
        idleTimeout = null;
    }

    function zoom() {
        var t = TheApp.draw.vvis.transition().duration(750);
        TheApp.draw.vvis.select(".axis--x").transition(t).call(xAxis);
        TheApp.draw.vvis.select(".axis--y").transition(t).call(yAxis);
        TheApp.draw.vvis.selectAll("circle").transition(t)
            .attr("cx", function(d) { return x(d.x); })
            .attr("cy", function(d) { return y(d.y); })
            .attr("r",  function(d) { return (Math.log(sc(1) /sc_size) + 1) * d.dot; })
            // .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ") scale(" + (Math.log(sc(1) /sc_size) + 1) + ")"; })
        ;
        textcontainer.selectAll(".scattertext").transition(t)
            .attr("transform", function(d) { return "translate(" + (x(d.x) + 6) + "," + (y(d.y) - 2) + ") scale(" + (Math.log(sc(1) /sc_size) + 1) + ")"; })
        ;
    }
};

TheApp.draw.drawCnnWeights = function(tokens, weights){
    TheApp.draw.cvis.selectAll("*").remove();
    var cnnField = $("#cnnWeightsField");
    var width = cnnField.find("svg").innerWidth(), height = cnnField.find("svg").innerHeight();
    width = width - TheApp.draw.cvismargin.left - TheApp.draw.cvismargin.right;
    height = height - TheApp.draw.cvismargin.top - TheApp.draw.cvismargin.bottom;

    var x = d3.scaleLinear()
        .range([10, width-10]);

    var step = height/((tokens.length + 2)*2);
    var y = d3.scalePoint()
        .domain(tokens)
        .range([step+10, height-step-10]);

    var xAxis = d3.axisBottom(x);

    var yAxis = d3.axisLeft(y);

    x.domain([0, weights[0].length]);

    TheApp.draw.cvis.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    TheApp.draw.cvis.append("g")
        .attr("class", "y axis")
        .call(yAxis);

    for(var j in tokens) {
        TheApp.draw.cvis.selectAll(".weight-line .token-" + j)
            .data(weights[j])
            .enter().append("line")
            .attr("class", "weight-line .token-" + j)
            .attr('x1', function (d, i) {
                return x(i);
            })
            .attr('y1', function (d, i) {
                return y(tokens[j]) - step;
            })
            .attr('x2', function (d, i) {
                return x(i);
            })
            .attr('y2', function (d, i) {
                return y(tokens[j]) + step;
            })
            .attr('style', function (d, i) {
                return "stroke: rgba(210,0,0, " + TheApp.draw.ValueToOpacity(d) + ")";
            });
    }
};

TheApp.draw.ValueToColor = function(d) {
    return Math.floor((1-TheApp.sigmoid(d*2))*255);
};

TheApp.draw.ValueToOpacity = function(d) {
    return  Math.max(Math.min(Math.pow(100, d)-0.5, 1.0), 0.0);
};

TheApp.sigmoid = function(x){
    return 1/(1+Math.exp(x));
};