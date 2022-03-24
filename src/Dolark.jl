module Dolark

    import YAML
    import Dolang
    import Dolo
    import OrderedCollections: OrderedDict    
    using StaticArrays

    # extracts annotated yaml structure from model file
    # the model file is supposed to contain two documents
    # - a valid dolo model (a.k.a. agent)
    # - a description of aggregate model
    function get_hmodel_source(txt::AbstractString)
        cons = YAML.Constructor()
        YAML.add_multi_constructor!((c,s,m)->m, cons, "tag:yaml.org")
        YAML.add_multi_constructor!((c,s,m)->m, cons, "!")
        data = YAML.load_all(txt, cons)
        agent, model = data
        return (;agent, model)
    end

    struct HModel{ID, AgentType}
        filename
        source
        symbols
        calibration
        exogenous
        equations
        factories
        agent::AgentType # dolo model 
    end

    import Base: show
    Base.show(io::IO, model::HModel) = print(io, "HModel")

    function get_symbols(yaml_node)
        symbols_src = yaml_node[:symbols]
        exogenous = [Symbol(e.value) for e in symbols_src[:exogenous]]
        aggregate = [Symbol(e.value) for e in symbols_src[:aggregate]]
        parameters = [Symbol(e.value) for e in symbols_src[:parameters]]
        return (;exogenous, aggregate, parameters)
    end

    function get_h_equations(yaml_node)
        projection = yaml_node[:projection].value
        equilibrium = yaml_node[:equilibrium].value
        return (;projection, equilibrium)
    end

    ### the next two functions are almost duplicates

    function gen_projection_ff(hmodel)
        projs_block = Dolang.parse_assignment_block(hmodel.equations.projection)
        
        eqs = OrderedDict{Symbol, Any}()
        for ch in projs_block.children
            lhs, rhs = ch.children
            name = Dolang.convert(Expr, lhs)
            eqs[name] = Dolang.convert(Expr, rhs)
        end
        
        arguments = OrderedDict{Symbol, Vector{Symbol}}()
        arguments[:y] = hmodel.symbols.aggregate
        arguments[:z] = hmodel.symbols.exogenous
        arguments[:p] = hmodel.symbols.parameters
        
        preamble = OrderedDict()

        return Dolang.FunctionFactory(eqs, arguments, preamble, :projection)
    end


    function gen_projection_ff(agent, symbols, equations)
        projs_block = Dolang.parse_assignment_block(equations.projection)
        
        eqs = OrderedDict{Symbol, Any}()
        for ch in projs_block.children
            lhs, rhs = ch.children
            name = Dolang.convert(Expr, lhs)
            eqs[name] = Dolang.convert(Expr, rhs)
        end
        
        arguments = OrderedDict{Symbol, Vector{Symbol}}()
        arguments[:y] = [Dolang.stringify(e,0) for e in symbols.aggregate]
        arguments[:z] = [Dolang.stringify(e,0) for e in symbols.exogenous]
        arguments[:p] = [Dolang.stringify(e) for e in symbols.parameters]
        
        preamble = OrderedDict()

        return Dolang.FunctionFactory(eqs, arguments, preamble, :projection)
    end

    import Dolang: Tree

    function gen_equilibrium_ff(agent, symbols, equations)

        eq_block = Dolang.parse_equation_block(equations.equilibrium)
        
        eqs = OrderedDict{Symbol, Any}()
        for (i,ch) in enumerate(eq_block.children)
            eq = Tree("sub", [ch.children[2], ch.children[1]]) 
            name = Symbol("out_", i, "_")
            eqs[name] = Dolang.convert(Expr, eq)
        end
        
        arguments = OrderedDict{Symbol, Vector{Symbol}}()
        arguments[:s] = [Dolang.stringify(e,0) for e in agent.symbols[:states]]
        arguments[:x] = [Dolang.stringify(e,0) for e in agent.symbols[:controls]]
        arguments[:y] = [Dolang.stringify(e,0) for e in symbols.aggregate]
        arguments[:z] = [Dolang.stringify(e,0) for e in symbols.exogenous]
        arguments[:Y] = [Dolang.stringify(e,1) for e in symbols.aggregate]
        arguments[:Z] = [Dolang.stringify(e,1) for e in symbols.exogenous]
        arguments[:p] = [Dolang.stringify(e) for e in symbols.parameters]
        preamble = OrderedDict()

        return Dolang.FunctionFactory(eqs, arguments, preamble, :equilibrium)

    end

    module ModelSpace

    end

    # import Dolark.ModelSpace

    function HModel(filename::AbstractString)
        
        txt = open(f->read(f, String), filename)

        source = get_hmodel_source(txt)
        symbols = get_symbols(source.model)
        calibration = Dolo.get_calibration(source.model)
        exogenous = Dolo.get_exogenous(source.model, symbols.exogenous, calibration.flat)
        equations = get_h_equations(source.model)
        

        agent = Dolo.Model(source.agent)
        AgentType = typeof(agent)
        ID = gensym()

        projection = gen_projection_ff(agent, symbols, equations)
        equilibrium = gen_equilibrium_ff(agent, symbols, equations)

        factories = (;projection, equilibrium)

        hmodel = HModel{ID, AgentType}(filename, source, symbols, calibration, exogenous, equations, factories, agent) 

        tm = typeof(hmodel)

        funp = Dolang.gen_generated_gufun(factories.projection; funname=:projection, dispatch=tm)
        fune = Dolang.gen_generated_gufun(factories.equilibrium; funname=:equilibrium, dispatch=tm)

        Core.eval(Dolark, funp)
        Core.eval(Dolark, fune)

        return hmodel

        
    end

    struct DModel
        hmodel
        F
        G
    end


    function discretize(hmodel::HModel, sol_agent)
        
        F = Dolo.Euler(hmodel.agent)
        G = Dolo.distG(hmodel.agent, sol_agent)

        DModel(hmodel, F, G)

    end

    struct Unknown
        μ
        p
        x
        y
    end


    using Dolang: _get_oorders

    include("temp.jl")

end # module
