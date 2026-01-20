{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.memex;
in {
  options.services.memex = {
    enable = lib.mkEnableOption "memex index service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.memex;
      description = "The memex package to use.";
    };

    continuous = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Run in continuous mode (watch) instead of interval mode.";
    };

    interval = lib.mkOption {
      type = lib.types.str;
      default = "3600";
      description = "Interval for indexing in seconds or systemd time span (if not continuous).";
    };

    watchInterval = lib.mkOption {
      type = lib.types.int;
      default = 30;
      description = "Polling interval in seconds (if continuous).";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.memex-index = {
      description = "Memex Index Service";
      wantedBy = ["default.target"];
      path = [cfg.package];
      serviceConfig = {
        ExecStart =
          if cfg.continuous
          then "${cfg.package}/bin/memex index --watch --watch-interval ${toString cfg.watchInterval}"
          else "${cfg.package}/bin/memex index";

        Restart =
          if cfg.continuous
          then "always"
          else "no";
        RestartSec = 10;
      };
    };

    systemd.user.timers.memex-index = lib.mkIf (!cfg.continuous) {
      description = "Memex Index Timer";
      wantedBy = ["timers.target"];
      timerConfig = {
        OnBootSec = "5m";
        OnUnitActiveSec = cfg.interval;
      };
    };
  };
}
